from csv_utils import get_classes_ordinal_from_config
import matplotlib.pyplot as plt
import constants as C
import os
import pandas as pd
import shutil
import tensorflow as tf
from utils.file_utils import get_filename_from_path
import tensorflow_hub as hub
import threading

from logging_cfg import get_logger
l = get_logger(__name__)


CLASS_ORDINAL_IDS = sorted(get_classes_ordinal_from_config())
NUMBER_OF_CLASSES = len(CLASS_ORDINAL_IDS)
ID_TO_INDEX_MAP = {id_: i for i, id_ in enumerate(CLASS_ORDINAL_IDS)}

class YamnetWrapper:
    _instance = None
    _lock = threading.Lock()
    SIG_SCORE = "output_0"

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(YamnetWrapper, cls).__new__(cls)
                cls._instance._model = None
                cls._instance._model_url = C.YAMNET_MODEL_URL
        return cls._instance

    def _load_model(self):
        """Lazy loads the TensorFlow Hub model when first accessed."""
        if self._model is None:
            l.info("Loading model initially...")
            self._model = hub.load(self._model_url)
            # class_map_path = self._model.class_map_path().numpy().decode('utf-8')
            # self.class_names = list(pd.read_csv(class_map_path)['display_name'])
            l.info("Model loaded.")

    def infer(self, inputs: tf.Tensor):
        """Runs inference using the Hub model."""
        self._load_model()
        # scores, embeddings, spectrogram
        return self._model(inputs)

    def extract_scores(self, input: tf.Tensor):
        self._load_model()
        scores, _, _ = self._model(input)
        return scores

    def extract_embedding(self, inputs: tf.Tensor):
        self._load_model()
        _, embeddings, _ = self._model(inputs)
        return embeddings
    
    def extract_spectrogram(self, input: tf.Tensor):
        self._load_model()
        _, _, spectrogram = self._model(input)
        return spectrogram

    def infer_score_class_name(self, inputs: tf.Tensor):
        """_summary_

        Args:
            inputs (tf.Tensor): _description_

        Returns:
            _type_: class_name: str, score: float
        """
        scores = self.extract_scores(inputs)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_class = tf.math.argmax(class_scores)
        top_score = class_scores[top_class].numpy()
        # return self.class_names[top_class], top_score
        return top_class, top_score


def plot_classname_distribution(df: pd.DataFrame):
    """
    Plot the distribution of class names in a dataframe
    """

    # Count the occurrences of each class
    class_counts = df[C.DF_CLASS_NAME_COL].value_counts()

    # Plot the bar chart
    plt.figure(figsize=(12, 6))
    class_counts.plot(kind="bar", color="skyblue", edgecolor="black")

    # Formatting the plot
    plt.title("Number of Data Points per Class", fontsize=14)
    plt.xlabel("Class Name", fontsize=12)
    plt.ylabel("Number of Data Points", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()

def copy_update_dataset_file(df: pd.DataFrame, dest_path: str) -> pd.DataFrame:
    """
    Copy dataset files from df to the given path\n
    Then, return a copy of the DataFrame with updated file paths.
    """
    cp_df = df.copy()
    os.makedirs(dest_path, exist_ok=True)
    shutil.rmtree(dest_path)
    os.makedirs(dest_path, exist_ok=True)
    
    # Copy files
    new_paths = []
    for old_path in cp_df["file_path"]:
        try:
            org_filename = get_filename_from_path(old_path)
            new_file_path = os.path.join(dest_path, org_filename)
            shutil.copy2(old_path, new_file_path)
            new_paths.append(new_file_path)
        except FileNotFoundError:
            new_paths.append("FILE_NOT_FOUND")

    # Update paths
    cp_df["file_path"] = new_paths
    return cp_df


def to_tensor_dataset(df: pd.DataFrame) -> tf.data.Dataset:

    """
    convert df to tensorflow compatible dataset     
    """

    from utils.wav_utils import load_wav_16k_mono_3
    
    def transform_wav(filepath: str, class_id: int, fold: int, filename):
        return load_wav_16k_mono_3(filepath), class_id, fold, filename
    
    filepaths = df[C.DF_PATH_COL]
    targets   = df[C.DF_CLASS_ID_COL]
    folds     = df[C.DF_FOLD_COL]
    filenames = df[C.DF_NAME_COL]
    

    ts_ds = tf.data.Dataset.from_tensor_slices((filepaths, targets, folds, filenames))
    return ts_ds.map(transform_wav)


def to_tensor_ds_embedding_extracted(dataset) -> tf.data.Dataset:
    if type(dataset) == pd.DataFrame:
        dataset = to_tensor_dataset(dataset)
    test_filenames = [
        "DOG_BARK_us8k_8214.wav",
        "DOG_BARK_esc50_1097.wav",
        "DOG_BARK_bdlib2_22.wav",
        "SIREN_us8k_530.wav",
        "SIREN_esc50_166.wav",
        "SIREN_esc50_348.wav",
        "THUNDER_STORM_bdlib2_68.wav",
        "THUNDER_STORM_esc50_1689.wav",
        "THUNDER_STORM_esc50_966.wav"
        
    ]
    sample_dataset_inspect(dataset, test_filenames)
    
    def extract_embedding(wav_data, label, fold, filename):
        ''' run YAMNet to extract embedding from the wav data '''
        yamnet = YamnetWrapper()
        embeddings = yamnet.extract_embedding(wav_data)
        num_embeddings = tf.shape(embeddings)[0]
        
        return (embeddings,
            tf.cast(tf.repeat(label, num_embeddings), tf.int16),
            tf.cast(tf.repeat(fold, num_embeddings), tf.int16),
            tf.repeat(filename, num_embeddings))
    
    # extract embedding
    return dataset.map(extract_embedding,
                    num_parallel_calls=tf.data.AUTOTUNE
                    ).unbatch()

def encode_label(embedding, label):
    """Convert numeric label ID to one-hot encoding."""

    label_index = ID_TO_INDEX_MAP[label.numpy()]  # Convert ID to index
    return embedding, tf.one_hot(label_index, depth=NUMBER_OF_CLASSES, dtype=tf.float32)

# Wrap it for tf.data compatibility
def encode_label_tf(embedding, label):
    return tf.py_function(func=encode_label, inp=[embedding, label], Tout=(tf.float32, tf.float32))

def count_dataset_size(dataset: tf.data.Dataset) -> int:
    return sum(1 for _ in dataset)


def sample_dataset_inspect(dataset: tf.data.Dataset, filter_filenames: list[str]) -> tf.data.Dataset:
    """
    Filter the dataset to only include elements whose 4th column (filename) is in the filter_filenames list.
    """
    l.info(f"Filtering dataset for filenames in provided list of length {len(filter_filenames)}.")
    if not filter_filenames:
        raise ValueError("filter_filenames list must not be empty.")

    # Convert the list to a tf.constant for efficient comparison
    filter_filenames_tf = tf.constant(filter_filenames)

    def filename_in_filter(*args):
        # args[-1] is the filename (4th column)
        return tf.reduce_any(tf.equal(args[3], filter_filenames_tf))

    filtered_dataset = dataset.filter(filename_in_filter)
    for elem in filtered_dataset:
        l.info(f"Filtered element: {elem}")

    return filtered_dataset

def select_random_filenames(dataset: pd.DataFrame, sample_size: int) -> tf.data.Dataset:
    """
    Select a random sample of elements from the dataset.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0.")

    count = dataset.shape[0]
    if sample_size > count:
        raise ValueError(f"sample_size {sample_size} is greater than dataset size {count}.")

    l.info(f"Sampling {sample_size} elements from dataset of size {count}.")
    
    # Shuffle the dataset and take the first `sample_size` elements
    sample_dataset = dataset.sample(n=sample_size, random_state=42)
    return sample_dataset[C.DF_NAME_COL].tolist()