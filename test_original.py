import os

from dframe_utils import to_tensor_ds_embedding_extracted
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import librosa

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav',
                                                'https://storage.googleapis.com/audioset/miaow_16k.wav',
                                                cache_dir='./',
                                                cache_subdir='test_data')

print(testing_wav_file_name)


# Utility functions for loading audio files and making sure the sample rate is correct.

@tf.function
def load_wav_16k_mono_3(filename):
    """Load a WAV file, convert it to a float tensor, and resample to 16 kHz single-channel audio using librosa."""
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)

    def resample_audio(wav, org_sr):
        resampled_wav = librosa.resample(y=wav, orig_sr=org_sr, target_sr=16000)
        return resampled_wav.astype(np.float32) 
    
    def tf_resample_audio(wav, org_sr):
        return tf.numpy_function(func=resample_audio, inp=[wav, org_sr], Tout=tf.float32)
    
    return tf_resample_audio(wav, sample_rate)


testing_wav_data = load_wav_16k_mono_3(testing_wav_file_name)

_ = plt.plot(testing_wav_data)

# Play the audio file.
# display.Audio(testing_wav_data, rate=16000)


class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names =list(pd.read_csv(class_map_path)['display_name'])

for name in class_names[:20]:
    print(name)
print('...')

scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
class_scores = tf.reduce_mean(scores, axis=0)
top_class = tf.math.argmax(class_scores)
inferred_class = class_names[top_class]

print(f'The main sound is: {inferred_class}')
print(f'The embeddings shape: {embeddings.shape}')


_ = tf.keras.utils.get_file('esc-50.zip',
                        'https://github.com/karoldvl/ESC-50/archive/master.zip',
                        cache_dir='./',
                        cache_subdir='datasets',
                        extract=True)


esc50_csv = './datasets/esc-50_extracted/ESC-50-master/meta/esc50.csv'
base_data_path = './datasets/esc-50_extracted/ESC-50-master/audio/'

pd_data = pd.read_csv(esc50_csv)
pd_data.head()


my_classes = ['dog', 'cat']
map_class_to_id = {'dog':0, 'cat':1}

filtered_pd = pd_data[pd_data.category.isin(my_classes)]

class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
filtered_pd = filtered_pd.assign(target=class_id)

full_path = filtered_pd['filename'].apply(lambda row: os.path.join(base_data_path, row))
filtered_pd = filtered_pd.assign(filename=full_path)

filtered_pd.head(10)

def to_tensor_dataset(df: pd.DataFrame) -> tf.data.Dataset:

    """
    convert df to tensorflow compatible dataset     
    """

    from utils.wav_utils import load_wav_16k_mono_3
    
    def transform_wav(filename: str, class_id, fold):
        return load_wav_16k_mono_3(filename), class_id, fold
    
    filenames = df['filename']
    targets = df['target']
    folds = df['fold']

    ts_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
    return ts_ds.map(transform_wav)

main_ds = to_tensor_dataset(filtered_pd)
main_ds.element_spec

main_ds = to_tensor_ds_embedding_extracted(main_ds)
main_ds.element_spec
# filenames = filtered_pd['filename']
# targets = filtered_pd['target']
# folds = filtered_pd['fold']

# main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
# main_ds.element_spec


# def load_wav_for_map(filename, label, fold):
#   return load_wav_16k_mono_3(filename), label, fold

# main_ds = main_ds.map(load_wav_for_map)
# main_ds.element_spec


# # applies the embedding extraction model to a wav data
# def extract_embedding(wav_data, label, fold):
#   ''' run YAMNet to extract embedding from the wav data '''
#   scores, embeddings, spectrogram = yamnet_model(wav_data)
#   num_embeddings = tf.shape(embeddings)[0]
#   return (embeddings,
#             tf.repeat(label, num_embeddings),
#             tf.repeat(fold, num_embeddings))

# # extract embedding
# main_ds = main_ds.map(extract_embedding).unbatch()
# main_ds.element_spec

cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 4)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)
test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)

train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)


my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,), dtype=tf.float32,
                        name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(my_classes))
], name='my_model')

my_model.summary()


my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=3,
                                            restore_best_weights=True)


history = my_model.fit(train_ds,
                       epochs=20,
                       validation_data=val_ds,
                       callbacks=callback)


loss, accuracy = my_model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
result = my_model(embeddings).numpy()

inferred_class = my_classes[result.mean(axis=0).argmax()]
print(f'The main sound is: {inferred_class}')


class ReduceMeanLayer(tf.keras.layers.Layer):
  def __init__(self, axis=0, **kwargs):
    super(ReduceMeanLayer, self).__init__(**kwargs)
    self.axis = axis

  def call(self, input):
    return tf.math.reduce_mean(input, axis=self.axis)


saved_model_path = './dogs_and_cats_yamnet'

input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle,
                                            trainable=False, name='yamnet')
_, embeddings_output, _ = embedding_extraction_layer(input_segment)
serving_outputs = my_model(embeddings_output)
serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
serving_model = tf.keras.Model(input_segment, serving_outputs)
serving_model.save(saved_model_path, include_optimizer=False)


tf.keras.utils.plot_model(serving_model)