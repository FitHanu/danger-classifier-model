"""
The project entry script for data processing & training
"""

from datetime import datetime, timedelta, timezone
import os
import numpy as np
import pandas as pd
import constants as C
import tensorflow as tf
import tensorflow_hub as hub
import traceback
from tensorflow import keras
from ds.dataset import PD_SCHEMA
from utils.json_utils import (init_default_class_name,
                              append_empty_mapping_to_config)
from utils.csv_utils import (read_csv_as_dataframe,
                            write_csv_meta)
from utils.dframe_utils import (to_tensor_ds_embedding_extracted,
                                count_dataset_size)
# from utils.date_utils import get_formated_date_as_string
# TODO: Fix this temporal
def get_formated_date_as_string():
    gmt_plus_7 = timezone(timedelta(hours=7))
    folder_name = datetime.now(gmt_plus_7).strftime("%Y-%m-%d-%H-%M")
    return folder_name
print(get_formated_date_as_string())

from utils.wav_utils import (convert_pcm_16_ffmpeg_pd_row,
                            convert_pcm_16_sox_pd_row,
                            force_convert_sox_pd_row,
                            force_convert_ffmpeg_pd_row,
                            validate_wav_pd_row)
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from utils.metric_utils import f1_score
from partition.split_tdt import split_tdt, init_cfg
from ds.esc50 import ESC50
from ds.us8k import UrbanSound8K
from ds.bdlib2 import BDLib2
from ds.gad import GAD


from logging_cfg import get_logger
l = get_logger(__name__)

# Params
USE_PROCESSED_DATASET = False
FORCE_WAV_CONVERTING  = False # If True, convert to correct .wav format ignoring check (SLOW)
PROCESS_DATA_ONLY     = False
TFLITE_MODEL_OPTIMIZE = False # If True, optimize the exported TFLite model

# Globe vars
TRAVIS_SCOTT = tf.data.AUTOTUNE

def workflow():
    """
    Main procedure
    """
    
    # Handle dataset processing
    
    if USE_PROCESSED_DATASET:
        ds_ts = get_cached_dataset()
    else:
        ds_ts = process_dataset()
    
    if PROCESS_DATA_ONLY:
        l.info("PROCESS_DATA_ONLY is set to true, skipping training...")
        exit(0)

    # Handle training
    train(ds_ts)


def get_cached_dataset() -> tf.data.Dataset:
    """
    Get cached dataset from the filtered augmented dataset csv file
    """
    l.info(f"Reading filtered augmented dataset from {C.FILTERED_AUG_FOLDED_META_CSV}")
    df = read_csv_as_dataframe(C.FILTERED_AUG_FOLDED_META_CSV)
    
    # Convert to tensor dataset
    ds_ts = to_tensor_ds_embedding_extracted(df)
    
    l.info(f"Dataset shape: {df.shape}")
    return ds_ts

def process_dataset() -> tf.data.Dataset:
    
    """
    dataset processing workflow
    """
    datasets_registry = [
        ESC50(),
        GAD(),
        UrbanSound8K(),
        BDLib2(),
    ]

    # Init paths, Default class names
    l.info(f"Creating empty dataset directory to {C.FILTERED_DATASET_PATH}")
    os.makedirs(C.FILTERED_DATASET_PATH, exist_ok=True)
    init_default_class_name()

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
            os.path.join(C.PROJECT_ROOT,"missing_files.csv"),  index=False
        )

    if not FORCE_WAV_CONVERTING:
        l.info(f"1st. Converting .wav files into PCM 16bit format inside {C.FILTERED_DATASET_PATH} using ffmpeg...")
        main_df.apply(convert_pcm_16_ffmpeg_pd_row, axis=1)
        
        l.info(f"2nd. Converting .wav files into PCM 16bit format inside {C.FILTERED_DATASET_PATH} using sox...")
        main_df.apply(convert_pcm_16_sox_pd_row, axis=1)

    # Filter invalid WAV files (not PCM)
    false_files = main_df[~main_df.apply(validate_wav_pd_row, axis=1)]
    
    l.info(f"Done validating and converting .wav files")
    
    if false_files.shape[0] > 0:
        l.warning(f"Theres still {false_files.shape[0]}/{main_df.shape[0]} invalid .wav files after conversion, dropping them...")
        main_df = main_df[~main_df.index.isin(false_files.index)]
        l.info(f"Dataset shape after dropping invalid .wav files: {main_df.shape}")

    # Save filtered dataframe to csv (before augmentation)
    filtered_meta = C.FILTERED_META_CSV
    l.info(f"Datasets filtering & converting done, saving meta file to {filtered_meta}")
    main_df.to_csv(filtered_meta, index=False)

    # Get split config
    cfg = init_cfg()
    l.info(f"Spliting with cfg {cfg.__str__()}")


    # Split
    aug_k_df = split_tdt(main_df, cfg)
    
    if FORCE_WAV_CONVERTING:
        l.info(f"Force converting .wav files into PCM 16bit format inside {C.FILTERED_DATASET_PATH} using sox...")
        aug_k_df.apply(force_convert_sox_pd_row, axis=1)
    
    
    # Save augmented dataframe to .csv
    final_meta = C.FILTERED_AUG_FOLDED_META_CSV
    l.info(f"Datasets processing done, saving meta file to {final_meta}")
    aug_k_df.to_csv(final_meta, index=False)


    # Convert to tf compatible dataset & return
    ds_ts = to_tensor_ds_embedding_extracted(aug_k_df)
    return ds_ts
    


def train(ds_ts: tf.data.Dataset) -> None:
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
    dataset_size = count_dataset_size(train_ds)
    
    train_ds = train_ds.map(lambda x, y: (tf.ensure_shape(x, (1024, )), tf.ensure_shape(y, (NUMBER_OF_CLASSES, ))))
    val_ds = val_ds.map(lambda x, y: (tf.ensure_shape(x, (1024, )), tf.ensure_shape(y, (NUMBER_OF_CLASSES, ))))
    test_ds = test_ds.map(lambda x, y: (tf.ensure_shape(x, (1024, )), tf.ensure_shape(y, (NUMBER_OF_CLASSES, ))))

    BATCH_SIZE = 16
    train_ds = train_ds.shuffle(min(1000, dataset_size)).cache().batch(BATCH_SIZE).prefetch(TRAVIS_SCOTT)
    val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(TRAVIS_SCOTT)
    test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(TRAVIS_SCOTT)
    
    
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
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), # For vectorized labels
        optimizer="adamax",
        metrics=[
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            f1_score
        ]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                patience=4,
                                                restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                                                monitor='val_loss',
                                                factor=0.5,
                                                patience=2)

    history = yamnet_tweaked.fit(train_ds,
                        epochs=200,
                        validation_data=val_ds,
                        callbacks=[
                            early_stop,
                            reduce_lr
                        ])
    
    # Save training history to log directory
    history_log_path = os.path.join(C.LOG_PATH, "training_history.txt")
    with open(history_log_path, "w") as f:
        for key, values in history.history.items():
            f.write(f"{key}: {values}\n")
    l.info(f"Training history saved to {history_log_path}")
    
    results = yamnet_tweaked.evaluate(test_ds, return_dict=True)
    
    loss      = results['loss']
    precision = results['precision']
    recall    = results['recall']
    f1        = results['f1_score']
    
    l.info(f"Final Loss:      {loss}")
    l.info(f"Final Precision: {precision}")
    l.info(f"Final Recall:    {recall}")
    l.info(f"Final F1 Score:  {f1}")

    the_model_path    = os.path.join(C.MODELS_PATH,
                                     get_formated_date_as_string(),
                                     C.MODELS_PATH)
    
    os.makedirs(the_model_path, exist_ok=True)
    
    saved_model_path  = os.path.join(the_model_path,
                                     "yamnet_tweaked")
    
    tflite_model_path = os.path.join(the_model_path,
                                     "yamnet_tweaked.tflite")
    
    os.makedirs(saved_model_path, exist_ok=True)
    
    # Define final model

    # 1st layer: input
    input_segment = tf.keras.layers.Input(shape=(15600,),
                                        dtype=tf.float32,
                                        batch_size=None,
                                        name='waveform_binary')

    # 2nd layer: yamnet_embedding_extraction - make it stateless
    embedding_extraction_layer = hub.KerasLayer(C.YAMNET_MODEL_URL,
                                                trainable=False,
                                                name='yamnet_embedding_extraction')

    class EmbeddingExtractionLayer(tf.keras.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.embedding_layer = embedding_extraction_layer
        
        def call(self, inputs):
            _, embeddings, _ = self.embedding_layer(tf.squeeze(inputs))
            return embeddings

    embeddings_output = EmbeddingExtractionLayer()(input_segment)

    # 3rd layer: yamnet_tweaked - make it stateless
    class YamnetTweakedLayer(tf.keras.Layer):
        def __init__(self, yamnet_model, **kwargs):
            super().__init__(**kwargs)
            self.yamnet_model = yamnet_model
        
        def call(self, inputs):
            return self.yamnet_model(inputs)

    yamnet_layer = YamnetTweakedLayer(yamnet_tweaked)
    serving_outputs = yamnet_layer(embeddings_output)

    # 4th layer: ReduceMeanLayer
    class ReduceMeanLayer(tf.keras.layers.Layer):
        def __init__(self, axis=0, **kwargs):
            super().__init__(**kwargs)
            self.axis = axis

        def call(self, input):
            return tf.math.reduce_mean(input, axis=self.axis, keepdims=True)

    reduced = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)

    # Final model
    serving_model = tf.keras.Model(input_segment, reduced)

    # Build the model with dummy data to initialize all variables
    dummy_input = tf.random.normal((1, 15600))
    _ = serving_model(dummy_input)

    l.info(f"Model summary:")
    serving_model.summary()

    # Create a concrete function that captures all variables
    @tf.function
    def serving_fn(waveform):
        waveform = tf.expand_dims(waveform, 0)
        result = serving_model(waveform, training=False)
        return result

    # Get concrete function
    concrete_fn = serving_fn.get_concrete_function(
        tf.TensorSpec(shape=[15600], dtype=tf.float32, name='waveform_binary')
    )
    
    # Freeze the model (replaces saved_model-based conversion)
    frozen_func = convert_variables_to_constants_v2(concrete_fn)
    frozen_func.graph.as_graph_def()
    
    # Optional debug print
    for node in frozen_func.graph.as_graph_def().node:
        if node.op == "ReadVariableOp":
            print(f"⚠️ Still contains ReadVariableOp: {node.name}")

    # Save with the concrete function
    # l.info(f"Saving model...")
    # tf.saved_model.save(
    #     serving_model,
    #     saved_model_path,
    #     signatures={'serving_default': concrete_fn}
    # )
    # l.info(f"Model saved to {saved_model_path}")

    # Convert to TFLite with special settings for variable handling
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func])

    if TFLITE_MODEL_OPTIMIZE:
        def representative_data_gen():
            for _ in range(100):
                sample = np.random.random((15600,)).astype(np.float32)
                yield [sample]
        
        converter.representative_dataset = representative_data_gen
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Only built-ins, no resource variables anymore
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = False  # Should be false now

    try:
        tflite_model = converter.convert()
    except Exception as e:
        l.error(f"First conversion attempt failed: {e}")
        # Try with different settings
        converter.experimental_enable_resource_variables = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    l.info(f"TFLite model saved to {tflite_model_path}")


def get_args():
    """
    Get arguments
    """
    import argparse

    parser = argparse.ArgumentParser(description="Workflow")
    parser.add_argument(
        "--clean_cache",
        help="Clean cached dataset processes",
        action="store_true"
    )
    parser.add_argument(
        "--use_processed",
        help="Use the filtered augmented dataset in ./dataset",
        action="store_true"
    )
    parser.add_argument(
        "--force_wav_convert",
        help="Convert to correct .wav format ignoring check (SLOW)",
        action="store_true"
    )
    parser.add_argument(
        "--process_data_only",
        help="Process dataset only, skip training",
        action="store_true"
    )
    parser.add_argument(
        "--tflite_optimize",
        help="Optimize the exported TFLite model",
        action="store_true"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.clean_cache == True:
        from utils.file_utils import clean_user_cache_dir
        l.info("Cleaning user cache dir ...")
        c_dir = clean_user_cache_dir()
        l.info(f"Contents in {c_dir} has been cleaned.")

    if args.use_processed:
        l.info("Using processed dataset, skipping dataset processing...")
        USE_PROCESSED_DATASET = True

    if args.force_wav_convert:
        l.info("Forcing .wav conversion, ignoring check...")
        FORCE_WAV_CONVERTING = True

    if args.process_data_only:
        l.info("Processing dataset only, training will be skipped...")
        PROCESS_DATA_ONLY = True

    if args.tflite_optimize:
        l.info("exported tflite model will be optimized!")
        TFLITE_MODEL_OPTIMIZE = True


    try:
        workflow()
    except Exception as e:
        l.error(f"Error while executing workflow: {e}")
        l.error(f"{traceback.print_exc()}")
        l.info(f"Exiting with code 1, full log saved to {C.LOG_PATH}")
        exit(1)
