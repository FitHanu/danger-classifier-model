import os
import constants as C
import numpy as np
import tensorflow as tf
from wav_utils import load_wav_16k_mono_3



def inference(model, wav_data):
    """
    Perform inference on the given model with the provided wav data.
    
    Args:
        model: The loaded TensorFlow model.
        wav_data: The input wav data for inference.
        
    Returns:
        The output of the model.
    """
    infer = model.signatures["serving_default"]
    result = infer(wav_data)
    return result

def main():
    saved_model_path = os.path.join(C.MODELS_PATH, "yamnet_tweaked")
    reloaded_model = tf.saved_model.load(saved_model_path)
    
    # sample_file = os.path.join(C.FILTERED_DATASET_PATH, "SIREN_us8k_5776.wav")
    #dataset/CAR_HORN_us8k_7693.wav
    sample_file = os.path.join(C.FILTERED_DATASET_PATH, "CAR_HORN_us8k_7693.wav")
    wav_data = load_wav_16k_mono_3(sample_file)
    infer = reloaded_model.signatures["serving_default"]
    result = infer(wav_data)
    # print(result)
    tensor_values = result["output_0"]
    max_score = tf.reduce_max(tensor_values)
    max_index = tf.argmax(tensor_values)
    print(f"Max Score: {max_score}, Index: {max_index}")

if __name__ == "__main__":
    main()