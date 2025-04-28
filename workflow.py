"""
The project entry script
"""

import os
import pandas as pd
import constants as C
import tensorflow as tf
import tensorflow_hub as hub
import traceback
from tensorflow import keras
from constants import PROJECT_ROOT
from ds.dataset import PD_SCHEMA
from utils.json_utils import init_default_class_name, append_empty_mapping_to_config
from utils.file_utils import init_class_folds, get_filename_without_extension
from utils.csv_utils import (get_classes_ordinal_from_config, read_csv_as_dataframe,
                            write_csv_meta,
                            get_classes_from_config)
from utils.dframe_utils import (NUMBER_OF_CLASSES, plot_classname_distribution,
                                copy_update_dataset_file,
                                to_tensor_ds_embedding_extracted)
from utils.wav_utils import (convert_pcm_pd_row,
                            convert_pcm_pd_row_2,
                            validate_wav_pd_row)
from utils.metric_utils import f1_score
from partition.split_tdt import split_tdt, init_cfg
from ds.esc50 import ESC50
from ds.us8k import UrbanSound8K
from ds.bdlib2 import BDLib2
from ds.gad import GAD

from logging_cfg import get_logger
l = get_logger(__name__)

TRAVIS_SCOTT = tf.data.AUTOTUNE
IS_TRAIN = false

def workflow():
    """
    Main procedure
    """
    datasets_registry = [
        ESC50(),
        GAD(),
        UrbanSound8K(),
        BDLib2(),
    ]

    # Init paths, Default class names
    # DATASET_PATH_FILTERED = os.path.join(PROJECT_ROOT, "dataset")
    l.info(f"Creating empty dataset directory to {C.FILTERED_DATASET_PATH}")
    os.makedirs(C.FILTERED_DATASET_PATH, exist_ok=True)
    init_default_class_name()
    # init_class_folds(DATASET_PATH_FILTERED)

    # Init main dataframe
    main_df = pd.DataFrame(columns=PD_SCHEMA.keys()).astype(PD_SCHEMA)

    # Process each dataset
    for ds in datasets_registry:
        # Add empty mapping to config (initial)
        l.info(f"Filtering & mapping class names for {ds.key}")
        append_empty_mapping_to_config(ds, overwrite=False)

        # Call ds life cycle methods
        ds.hell_yeah()
        l.info(f'Dataset "{ds.name}" info saved to {C.FILTERED_DATASET_PATH}')
        # Read filtered metafile
        df = read_csv_as_dataframe(ds.get_filtered_meta_path())

        # Append to main dataframe
        main_df = pd.concat([main_df, df], ignore_index=True)
        l.info(f"main_df shape after filter: {main_df.shape}")

    l.info(f"Done filtering & mapping class names for all datasets")
    l.info(f"Main dataframe shape: {main_df.shape}")

    # Write main dataframe to csv

    l.info(f"Writing filtered merged meta file into: {C.MERGED_META_CSV}")
    write_csv_meta(main_df, "merged")
    
    l.info(f"Validating .wav files from merged dataset, path: {C.FILTERED_DATASET_PATH}")
    
    # Count missing files after filtering
    missing_files = main_df[main_df[C.DF_PATH_COL].apply(os.path.isfile)]
    if missing_files.shape[0] > 0:
        l.warning(f"Missing files: {missing_files.shape[0]}")
        missing_files.to_csv(
            C.PROJECT_ROOT + os.path.sep + "missing_files.csv", index=False
        )

    # l.info(f"1st. Converting .wav files into PCM 16bit format inside {C.FILTERED_DATASET_PATH} using ffmpeg...")
    # main_df.apply(convert_pcm_pd_row, axis=1)

    l.info(f"2nd. Converting .wav files into PCM 16bit format inside {C.FILTERED_DATASET_PATH} using sox...")
    main_df.apply(convert_pcm_pd_row_2, axis=1)

    # Filter invalid WAV files (not PCM)
    false_files = main_df[~main_df.apply(validate_wav_pd_row, axis=1)]
    
    l.info(f"Done validating and converting .wav files")
    
    if false_files.shape[0] > 0:
        l.warning(f"Theres still {false_files.shape[0]}/{main_df.shape[0]} invalid .wav files after conversion, dropping them...")
        main_df = main_df[~main_df.index.isin(false_files.index)]
        l.info(f"Dataset shape after dropping invalid .wav files: {main_df.shape}")



    # Get split config
    cfg = init_cfg()
    l.info(f"Spliting with cfg {cfg.__str__()}")


    # Split
    aug_k_df = split_tdt(main_df, cfg)
    
    
    # Save augmented dataframe to .csv
    final_meta = C.FILTERED_AUG_FOLDED_META_CSV
    l.info(f"Datasets processing done, saving meta file to {final_meta}")
    aug_k_df.to_csv(final_meta, index=False)
    ds_ts = to_tensor_ds_embedding_extracted(aug_k_df)
    
    
    # Filter train, val, test by fold label
    cached_ds = ds_ts.cache()
    train_ds = cached_ds.filter(lambda embedding, class_name, fold: fold < 8)
    val_ds = cached_ds.filter(lambda embedding, class_name, fold: fold == 8)
    test_ds = cached_ds.filter(lambda embedding, class_name, fold: fold == 9)
    
    
    # Remove fold column
    remove_fold_column = lambda embedding, class_name, fold: (embedding, class_name)
    train_ds = train_ds.map(remove_fold_column)
    val_ds = val_ds.map(remove_fold_column)
    test_ds = test_ds.map(remove_fold_column)
    
    
    # One hot encoding for labels
    from utils.dframe_utils import encode_label_tf, NUMBER_OF_CLASSES
    train_ds = train_ds.map(encode_label_tf, num_parallel_calls=TRAVIS_SCOTT)
    val_ds = val_ds.map(encode_label_tf, num_parallel_calls=TRAVIS_SCOTT)
    test_ds = test_ds.map(encode_label_tf, num_parallel_calls=TRAVIS_SCOTT)
    
    
    # Batching and shuffling
    def count_dataset_size(dataset):
        return sum(1 for _ in dataset)
    dataset_size = count_dataset_size(train_ds)
    
    train_ds = train_ds.map(lambda x, y: (tf.ensure_shape(x, (1024, )), tf.ensure_shape(y, (NUMBER_OF_CLASSES, ))))
    val_ds = val_ds.map(lambda x, y: (tf.ensure_shape(x, (1024, )), tf.ensure_shape(y, (NUMBER_OF_CLASSES, ))))
    test_ds = test_ds.map(lambda x, y: (tf.ensure_shape(x, (1024, )), tf.ensure_shape(y, (NUMBER_OF_CLASSES, ))))

    BATCH_SIZE = 16
    train_ds = train_ds.shuffle(min(1000, dataset_size)).cache().batch(BATCH_SIZE).prefetch(TRAVIS_SCOTT)
    val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(TRAVIS_SCOTT)
    test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(TRAVIS_SCOTT)
    
    
    # Model setup
    # yamnet_tweaked = tf.keras.Sequential([
    #     tf.keras.layers.Input(shape=(1024,), batch_size=None, dtype=tf.float32, name='input_embedding'),  
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     # Add GAP1D layer to reduce the dimensionality (None part of the shape=(None, 1024))
    #     # Make the model dimension independent
    #     # tf.keras.layers.GlobalAveragePooling1D(),
    #     tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax', name="class_scores")  # Output class probabilities
    # ], name='yamnet_tweaked')
    inputs = tf.keras.layers.Input(shape=(1024,), dtype=tf.float32, name='input_embedding')

    # Hidden layer
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)

    # Output layer
    outputs = tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax', name="class_scores")(x)

    # Define Functional model
    yamnet_tweaked = tf.keras.Model(inputs=inputs, outputs=outputs, name='yamnet_tweaked')

    yamnet_tweaked.summary()
    
    # Compile the model
    yamnet_tweaked.compile(
        # # raw scores (logits) instead of probabilities (if the final layer doesn’t have softmax).
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(), # For non vectorized labels
        loss=tf.keras.losses.CategoricalCrossentropy(), # For vectorized labels
        optimizer="adamax",
        metrics=[
            keras.metrics.Precision(name="precision"),  
            keras.metrics.Recall(name="recall"),
            f1_score
        ]
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                patience=4,
                                                restore_best_weights=True)

    history = yamnet_tweaked.fit(train_ds,
                        epochs=200,
                        validation_data=val_ds,
                        callbacks=callback)
    
    # Save training history to log directory
    history_log_path = os.path.join(C.LOG_PATH, "training_history.txt")
    with open(history_log_path, "w") as f:
        for key, values in history.history.items():
            f.write(f"{key}: {values}\n")
    l.info(f"Training history saved to {history_log_path}")
    
    results = yamnet_tweaked.evaluate(test_ds, return_dict=True)
    
    loss = results['loss']
    precision = results['precision']
    recall = results['recall']
    f1 = results['f1_score']
    
    l.info(f"Final Loss: {loss}")
    l.info(f"Final Precision: {precision}")
    l.info(f"Final Recall: {recall}")
    l.info(f"Final F1 Score: {f1}")

    saved_model_path = os.path.join(C.MODELS_PATH, "yamnet_tweaked")
    tflite_model_path = os.path.join(C.MODELS_PATH, "yamnet_tweaked.tflite")
    
    os.makedirs(saved_model_path, exist_ok=True)
    # Define final model
    
    # 1st layer: input
    input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
    
    # 2nd layer: yamnet_embedding_extraction
    embedding_extraction_layer = hub.KerasLayer(C.YAMNET_MODEL_URL,
                                                trainable=False,
                                                name='yamnet_embedding_extraction')
    class EmbeddingExtractionLayer(tf.keras.Layer):
        def call(self, inputs):
            # scores, embeddings, spectrogram
            _, embeddings, _ = embedding_extraction_layer(inputs)
            return embeddings
    
    embeddings_output = EmbeddingExtractionLayer()(input_segment)
    
    # 3rd layer: yamnet_tweaked - on top of 2nd layer
    serving_outputs = yamnet_tweaked(embeddings_output)
    
    # 4th layer: ReduceMeanLayer - on top of 3rd layer
        # Define the final final layer
    class ReduceMeanLayer(tf.keras.layers.Layer):
        def __init__(self, axis=0, **kwargs):
            super(ReduceMeanLayer, self).__init__(**kwargs)
            self.axis = axis

        def call(self, input):
            return tf.math.reduce_mean(input, axis=self.axis)
    serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
    
    # Specify input_segment as the start layer of the sequential
    serving_model = tf.keras.Model(input_segment, serving_outputs)
    l.info(f"Model summary:")
    serving_model.summary()
    l.info(f"Saving model...")
    serving_model.export(saved_model_path, include_optimizer=False)
    l.info(f"Model saved to {saved_model_path}")
    
    
    # Convert to TFLite
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)



def get_args():
    """
    Get arguments
    """
    import argparse

    parser = argparse.ArgumentParser(description="Workflow")
    parser.add_argument(
        "--clean-cache", help="Clean cached dataset processes", action="store_true"
    )
    parser.add_argument(
        "--train", help="Do training, else process data only", action="store_true"
    )
    return parser.parse_args()


def test():
    pth = os.path.join(PROJECT_ROOT, "dataset", "merged.augmented.folded.csv")
    ds = pd.read_csv(pth)
    ds_ts = to_tensor_ds_embedding_extracted(ds)
    
    def count_dataset_size(dataset):
        return sum(1 for _ in dataset)
    
    dataset_size = count_dataset_size(ds_ts)

    print("Sample after embedding extracted: ")
    for x, y, z in ds_ts.shuffle(min(1000, dataset_size)).cache().take(1):
        print(x.shape, y.shape, z.shape)
        print(x)
        print(y)
    
    cached_ds = ds_ts.cache()
    train_ds = cached_ds.filter(lambda embedding, class_name, fold: fold < 8)
    val_ds = cached_ds.filter(lambda embedding, class_name, fold: fold == 8)
    test_ds = cached_ds.filter(lambda embedding, class_name, fold: fold == 9)
    
    # Remove fold column
    remove_fold_column = lambda embedding, class_name, fold: (embedding, class_name)
    train_ds = train_ds.map(remove_fold_column)
    val_ds = val_ds.map(remove_fold_column)
    test_ds = test_ds.map(remove_fold_column)
    
    print("Sample after remove fold: ")
    for x, y in train_ds.shuffle(min(1000, dataset_size)).cache().take(1):
        print(x.shape, y.shape)
        print(x)
        print(y)
    
    from dframe_utils import encode_label_tf
    train_ds = train_ds.map(encode_label_tf, num_parallel_calls=TRAVIS_SCOTT)
    val_ds = val_ds.map(encode_label_tf, num_parallel_calls=TRAVIS_SCOTT)
    test_ds = test_ds.map(encode_label_tf, num_parallel_calls=TRAVIS_SCOTT)


    print(f"Dataset size: {dataset_size}")
    print("Sample after one hot encoding: ")
    for x, y in train_ds.shuffle(min(1000, dataset_size)).cache().take(1):
        print(x.shape, y.shape)
        print(x)
        print(y)
        
    train_ds = train_ds.map(lambda x, y: (tf.ensure_shape(x, (1024, )), tf.ensure_shape(y, (NUMBER_OF_CLASSES, ))))
    val_ds = val_ds.map(lambda x, y: (tf.ensure_shape(x, (1024, )), tf.ensure_shape(y, (NUMBER_OF_CLASSES, ))))
    test_ds = test_ds.map(lambda x, y: (tf.ensure_shape(x, (1024, )), tf.ensure_shape(y, (NUMBER_OF_CLASSES, ))))

    BATCH_SIZE = 16
    train_ds = train_ds.shuffle(min(1000, dataset_size)).cache().batch(BATCH_SIZE).prefetch(TRAVIS_SCOTT)
    val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(TRAVIS_SCOTT)
    test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(TRAVIS_SCOTT)
    
    print("Sample after batching and shuffling: ")
    for x, y in train_ds.take(1):
        print(x.shape, y.shape)
        print(x)
        print(y)

if __name__ == "__main__":
    args = get_args()
    if args.train == True:
        IS_TRAIN == True
    if args.clean_cache == True:
        from utils.file_utils import clean_user_cache_dir
        l.info("Cleaning user cache dir ...")
        c_dir = clean_user_cache_dir()
        l.info(f"Contents in {c_dir} has been cleaned.")
    try:
        workflow()
    except Exception as e:
        l.error(f"Error while executing workflow: {e}")
        l.error(f"{traceback.print_exc()}")
        l.info(f"Exiting with code 1, full log saved to {C.LOG_PATH}")
        exit(1)
    # test()

