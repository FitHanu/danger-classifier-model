import os
import subprocess
import wave
import tensorflow as tf
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional
import traceback
import struct
import constants as C

from logging_cfg import get_logger
l = get_logger(__name__)

@dataclass
class WavInfo:
    sample_rate: int
    num_channels: int
    duration: float
    audio_format: str

class InvalidArgument(Exception):
    pass


class KnownWavFormat:
    """
        #WAVE_FORMAT_GSM610 - 49
        #WAVE_FORMAT_PCM - 1
        #WAVE_FORMAT_EXTENSIBLE - 65534
        #WAVE_IEEE_FLOAT - 3
    """
    WAVE_FORMAT_GSM610 = 0x0031
    WAVE_FORMAT_PCM = 0x0001
    WAVE_FORMAT_EXTENSIBLE = 0xFFFE
    WAVE_IEEE_FLOAT = 0x0003
    WAVE_UNKNOWN = -1
    as_set = {
            WAVE_FORMAT_GSM610: "WAVE_FORMAT_GSM610",
            WAVE_FORMAT_PCM: "WAVE_FORMAT_PCM",
            WAVE_FORMAT_EXTENSIBLE: "WAVE_FORMAT_EXTENSIBLE",
            WAVE_IEEE_FLOAT: "WAVE_IEEE_FLOAT",
            WAVE_UNKNOWN: "WAVE_UNKNOWN"
            }



def get_wav_data_length(file_path: str) -> float:
    # Validate file path
    if not os.path.isfile(file_path):
        raise InvalidArgument(f"The provided path is not a valid file: {file_path}")
    # Validate file type
    if not file_path.lower().endswith('.wav'):
        raise InvalidArgument(f"The provided file is not a WAV file: {file_path}")
    
    # Read WAV file and calculate length in milliseconds
    try:
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration * 1000  # Convert to milliseconds
    except Exception as e:
        print(f"Error processing WAV file: {file_path}, error: {e}")
        raise e

def convert_to_pcm_replace_ffmpeg(input_file:str) -> None:
    """
    Converts a WAV file from WAVE_FORMAT_EXTENSIBLE or WAVE_IEEE_FLOAT to WAVE_FORMAT_PCM
    and replaces the original file.
    
    Args:
        input_file (str): Path to the input WAV file.
    """
    # Create a temporary output file
    temp_file = input_file + ".tmp.wav"
    sussy_message = [
        "Unsupported codec",
        "Invalid channel layout",
        "Invalid sample format"
    ]
    # FFmpeg command to convert to PCM (signed 16-bit little-endian)
    command = [
        "ffmpeg", "-y",  # Overwrite existing file
        "-i", input_file,  # Input file
        "-acodec", "pcm_s16le",  # Convert to PCM 16-bit
        temp_file  # Output temporary file
    ]

    try:
        # Run FFmpeg conversion
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Capture FFmpeg output
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # Check if any suspicious message is in the output
        for message in sussy_message:
            if message in result.stderr:
                l.warning(f"Error might be occurred while processing file {input_file}, error: \n {result.stderr}")
                break

        # Replace original file
        os.replace(temp_file, input_file)
        l.info(f"Converted and replaced: {input_file}")
        
    except subprocess.CalledProcessError as e:
        l.error(f"Error converting {input_file}: {e}")

def convert_to_pcm_replace_sox(input_file: str) -> None:
    """
    Converts a WAV file from WAVE_FORMAT_EXTENSIBLE or WAVE_IEEE_FLOAT to WAVE_FORMAT_PCM
    and replaces the original file using SoX.

    Args:
        input_file (str): Path to the input WAV file.
    """
    temp_file = input_file + ".tmp.wav"

    # SoX command to convert to PCM (signed 16-bit little-endian)
    command = [
        "sox",
        input_file,
        "-e", "signed-integer",
        "-b", "16", # bit depth
        # "-c", "1", # chanel
        # "-r", "44100", # sample rate
        temp_file, 
    ]

    try:
        # Run SoX conversion
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Check for potential warnings/errors
        if result.stderr:
            l.warning(f"Warnings or errors encountered while processing {input_file}:\n{result.stderr}")

        # Replace original file
        os.replace(temp_file, input_file)
        l.info(f"Converted and replaced: {input_file}")
        
    except subprocess.CalledProcessError as e:
        l.error(f"Error converting {input_file}: {e}")


def get_wave_data_length_2(row: pd.Series) -> float:
    """
    For large and possibly falsy dataset, `return -1` for defect .wav file for further processing
    """
    # Validate file path
    try:
        file_path = row[C.DF_PATH_COL]
        with sf.SoundFile(file_path) as audio_file:
            len = len(audio_file)
            sr = audio_file.samplerate
            duration = len / sr
            return duration * 1000  # Convert to milliseconds
        row[C.DF_LENGTH_COL] = file_length
    except Exception as e:
        l.warning(f"Could not get .wav length for {row[C.DF_PATH_COL]}, {e}")
        traceback.print_exc()
        row[C.DF_LENGTH_COL] = -1
    return row

def load_wav(filename):
    file_contents = tf.io.read_file(filename)
    wav, _ = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    return wav

def load_wav_mono(filename):
    wav = load_wav(filename)
    return tf.squeeze(wav, axis=-1)

@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def plot_mono_wav(wav, title="Waveform", sample_rate=16000, figname="plot.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(wav.numpy(), label="Audio Waveform")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    plt.legend()
    plt.savefig(figname)

@tf.function
def load_wav_16k_mono_2(filename):
    """Load a WAV file, convert it to a float tensor, and resample using tf.signal.resample."""
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)


    target_length = tf.cast(tf.shape(wav)[0] * 16000 / tf.cast(sample_rate, tf.float32), tf.int32)
    wav_resampled = tf.signal.resample(wav, target_length)

    return wav_resampled

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


def get_wave_format(file_path: str) -> KnownWavFormat:
    """_summary_
    Get the format of a WAV file. The function reads the file header to determine the format code.
    Args:
        file_path (str): _description_

    Returns:
        KnownWavFormat: _description_
    """
    try:
        with open(file_path, "rb") as f:
            f.seek(20)  # Seek to the format code position
            format_code = struct.unpack("<H", f.read(2))[0]

        match format_code:
            # 0x0001
            case KnownWavFormat.WAVE_FORMAT_PCM:
                return KnownWavFormat.WAVE_FORMAT_PCM
            # 0xFFFE
            case KnownWavFormat.WAVE_FORMAT_EXTENSIBLE:
                return KnownWavFormat.WAVE_FORMAT_EXTENSIBLE
            # 0x0031
            case KnownWavFormat.WAVE_FORMAT_GSM610:
                return KnownWavFormat.WAVE_FORMAT_GSM610
            # 0x0003
            case KnownWavFormat.WAVE_IEEE_FLOAT:
                return KnownWavFormat.WAVE_IEEE_FLOAT
            # Unknown format
            case _:
                l.warning(f"{file_path} has an unknown WAV format: {format_code}")
                return KnownWavFormat.WAVE_UNKNOWN

    except Exception as e:
        l.error(f"Error validating WAV file: {file_path}, error: {e}")
        return KnownWavFormat.WAVE_UNKNOWN

def validate_wav_pcm_format(file_path) -> bool:
    """_summary_
    Return False if the file is not in PCM format.
    Args:
        file_path (_type_): _description_

    Returns:
        bool: _description_
    """
    try:
        if get_wave_format(file_path) != KnownWavFormat.WAVE_FORMAT_PCM:
            return False
    except Exception as e:
        l.warning(f"Error occur when validating WAV file: {file_path}, error: {e}")
        return False
    return True


    
def convert_pcm_pd_row(row: pd.Series) -> None:
    file_path = row[C.DF_PATH_COL]
    if not validate_wav_pcm_format(file_path):
        convert_to_pcm_replace_ffmpeg(file_path)

def convert_pcm_pd_row_2(row: pd.Series) -> None:
    file_path = row[C.DF_PATH_COL] 
    if not validate_wav_pcm_format(file_path):
        convert_to_pcm_replace_sox(file_path)

def validate_wav_pd_row(row: pd.Series) -> bool:
    try:
        return validate_wav_pcm_format(row[C.DF_PATH_COL])
    except Exception as e:
        l.warning(f"Error validating WAV file: {row[C.DF_PATH_COL]}, error: {e}")
        return False

# if __name__ == "__main__":
    # path = "/workspaces/sound-recognition-ai/python/ds/meta/merged.csv"
    # import pandas as pd
    # import constants as C
    
    # false_files = pd.DataFrame()
    
    # def validate_wav(row: pd.Series) -> bool:
    #     global false_files
    #     try:
    #         file_path = row[C.DF_PATH_COL]
    #         valid = validate_wav_format(file_path)
    #         if not valid:
    #             false_files = pd.concat([false_files, pd.DataFrame([row])], ignore_index=True)
    #     except Exception as e:
    #         l.warning(f"Error validating WAV file: {file_path}, error: {e}")
    #         return False
    # def reformat(row: pd.Series):
    #     file_path = row[C.DF_PATH_COL] 
    #     if not validate_wav_format(file_path):
    #         l.info(f"Converting {file_path} to PCM")
    #         convert_to_pcm_replace_ffmpeg(file_path)
    
    # def reformat_2(row: pd.Series):
    #     file_path = row[C.DF_PATH_COL] 
    #     if not validate_wav_format(file_path):
    #         l.info(f"Converting {file_path} to PCM")
    #         convert_to_pcm_replace_sox(file_path)

    # df = pd.read_csv(path) 
    # df.apply(reformat, axis=1)
    # df.apply(reformat_2, axis=1)
    # df.apply(validate_wav, axis=1)

    # print(len(false_files))
    # print(false_files.head(10))


# def main():
#     testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav',
#                                                 'https://storage.googleapis.com/audioset/miaow_16k.wav',
#                                                 cache_dir='./',
#                                                 cache_subdir='test_data')
#     # wav = load_wav_mono(testing_wav_file_name)
#     wav0 = load_wav(testing_wav_file_name)
#     wav1 = load_wav_mono(testing_wav_file_name)
#     wav2 = load_wav_16k_mono_3(testing_wav_file_name)
#     print(get_wav_info(testing_wav_file_name))
#     print(get_wav_info_from_tensor(wav0))
#     print(get_wav_info_from_tensor(wav1))
#     print(get_wav_info_from_tensor(wav2))
#     os.makedirs("python/plots", exist_ok=True)
#     plot_mono_wav(wav0, figname="python/plots/plot0.png")
#     plot_mono_wav(wav1, figname="python/plots/plot1.png")
#     plot_mono_wav(wav2, figname="python/plots/plot2.png")


# if __name__ == "__main__":
#     main()
