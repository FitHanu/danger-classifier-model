import os
import kagglehub
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import constants as C
from dframe_utils import YamnetWrapper
from wav_utils import load_wav_16k_mono_3


def plot_tf_float32_converted_from_mono_wav(
    file_path: str,
    output_path: str,
) -> None:
    """
    Plot a waveform from a tf.float32 tensor.

    Args:
    """
    waveform = load_wav_16k_mono_3(file_path)
    # Create time axis
    time = np.linspace(0, len(waveform), num=len(waveform))

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, waveform, label="Audio Waveform", color="blue")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Plot of Mono WAV Tensor (tf.float32)")

    plt.savefig(output_path)


def plot_spectrogram_wav(
    file_path: str,
    output_path: str,
) -> None:
    """
    Plot a spectrogram from a WAV file.

    Args:
    """
    yamnet = YamnetWrapper()
    yamnet._load_model()

    audio_tensor = load_wav_16k_mono_3(file_path)
    spectrogram = yamnet.extract_spectrogram(audio_tensor)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram.numpy().T, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.savefig(output_path)


def plot_embedding_extracted(
    file_path: str,
    output_path: str,
) -> None:
    """
    Plot a spectrogram from a WAV file.

    Args:
    """
    yamnet = YamnetWrapper()
    yamnet._load_model()

    audio_tensor = load_wav_16k_mono_3(file_path)
    embedding = yamnet.extract_embedding(audio_tensor)

    print(f"Embedding shape: {embedding.shape}")

    # Plot the spectrogram
    # plt.figure(figsize=(10, 4))
    # plt.imshow(embedding.numpy().T, aspect='auto', origin='lower', cmap='viridis')
    # # plt.colorbar(format='%+2.0f dB')
    # plt.title("Embedding")
    # plt.ylabel("Embedding dimension")
    # plt.xlabel("Number of samples")
    plt.figure(figsize=(16, 6))
    plt.imshow(
        embedding.numpy().T, aspect="auto", origin="lower", cmap="plasma"
    )  # or cmap='viridis'
    y_start = 0
    y_end = embedding.shape[1]
    y_mid = y_end // 2
    plt.yticks([y_start, y_mid, y_end])
    plt.xticks(
        ticks=np.arange(embedding.shape[0]), labels=np.arange(embedding.shape[0])
    )
    plt.ylabel("Feature Index (1024-D)")
    plt.xlabel("Time Frames")
    plt.title("YAMNet Embeddings Heatmap")

    plt.savefig(output_path)



def down_1():
    path = kagglehub.dataset_download("chibuzokelechi/explosion-sound")
    print(path)
    return path

def test_1():
    
    def get_file_name_without_extension(file_name: str) -> str:
        return os.path.splitext(file_name)[0]

    point_map = [
        "ALARM_bdlib2_1.wav",
        "CAR_HORN_us8k_3703.wav",
        "DOG_BARK_bdlib2_2.wav",
        "GUNSHOT_HANDGUN_gad_373.wav",
        "CRYING_BABY_esc50_1728.wav",
        "THUNDER_STORM_esc50_1287.wav",
        "SIREN_us8k_7739.wav",
        "RAIN_esc50_1404.wav",
        "GUNSHOT_RIFLE_gad_145.wav",
        "CHAINSAW_esc50_861.wav",
        "GLASS_BREAKING_esc50_595.wav"
    ]
    

    
    path = down_1()
    extended_sound = [
        ["explosion", path, "336007__rudmer_rotteveel__multiple-deep-explosions-noisy-rec.wav"],
        # ["car_crash", C.FILTERED_DATASET_PATH, ""],
        # ["scream"]
    ]
    
    # req: explosion, car crash, scream

    for path in point_map:
        bare_name = get_file_name_without_extension(path)
        original_file_path = os.path.join("dataset", path)
        output_plot_wav_path = os.path.join(
            C.PROJECT_ROOT, "plots", "new", f"{bare_name}_waveform.png"
        )
        output_plot_spec_path = os.path.join(
            C.PROJECT_ROOT, "plots", "new", f"{bare_name}_spectrogram.png"
        )
        output_copy_wav_path = os.path.join(
            C.PROJECT_ROOT, "plots", "new", f"{bare_name}.copied.wav"
        )
        output_embedding_path = os.path.join(
            C.PROJECT_ROOT, "plots", "new", f"{bare_name}_embedding.png"
        )

        os.makedirs("plots", exist_ok=True)
        # Plot spectrogram lấy được từ Yamnet
        plot_spectrogram_wav(original_file_path, output_plot_spec_path)
        # Plot dạng cuối cùng của data trước khi cho vào train
        plot_tf_float32_converted_from_mono_wav(
            original_file_path, output_plot_wav_path
        )

        plot_embedding_extracted(original_file_path, output_embedding_path)
        # Copy file
        os.system(f"cp {original_file_path} {output_copy_wav_path}")
        
    new_1_path = os.path.join(C.PROJECT_ROOT, "plots", "new_1")
    for path in extended_sound:
        bare_name = path[0]
        original_file_path = os.path.join(path[1], path[2])
        output_plot_wav_path = os.path.join(
            new_1_path, f"{bare_name}_waveform.png"
        )
        output_plot_spec_path = os.path.join(
            new_1_path, f"{bare_name}_spectrogram.png"
        )
        output_copy_wav_path = os.path.join(
            new_1_path, f"{bare_name}.copied.wav"
        )
        output_embedding_path = os.path.join(
            new_1_path, f"{bare_name}_embedding.png"
        )

        os.makedirs("plots", exist_ok=True)
        os.makedirs("plots/new_1", exist_ok=True)
        # Plot spectrogram lấy được từ Yamnet
        plot_spectrogram_wav(original_file_path, output_plot_spec_path)
        # Plot dạng cuối cùng của data trước khi cho vào train
        plot_tf_float32_converted_from_mono_wav(
            original_file_path, output_plot_wav_path
        )

        plot_embedding_extracted(original_file_path, output_embedding_path)
        # Copy file
        os.system(f"cp {original_file_path} {output_copy_wav_path}")
        

if __name__ == "__main__":
    down_1()
    test_1()