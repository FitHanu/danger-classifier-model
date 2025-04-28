import pandas as pd
import constants as C
import os
import tensorflow as tf
from wav_utils import load_wav_16k_mono_3
from pathlib import Path


CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent

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
    dataset = pd.read_csv(C.FILTERED_AUG_FOLDED_META_CSV)
    test_fold = dataset[dataset[C.DF_FOLD_COL] == 9].copy()
    
    saved_model_path = os.path.join(C.MODELS_PATH, "yamnet_tweaked")
    reloaded_model = tf.saved_model.load(saved_model_path)
    
    # Create a new empty list to hold scores
    scores = []
    class_indices = []

    for idx, row in test_fold.iterrows():
        try:
            # Build the full path to the wav file
            file_path = os.path.join(C.FILTERED_DATASET_PATH, row[C.DF_PATH_COL])
            
            # Load WAV data
            wav_data = load_wav_16k_mono_3(file_path)
            
            # Perform inference
            result = inference(reloaded_model, wav_data)
            
            # Get the output tensor
            tensor_values = result["output_0"]
            
            # Extract the maximum score
            max_score = tf.reduce_max(tensor_values)
            max_index = tf.argmax(tensor_values)

            # Save the score
            scores.append(max_score.numpy())
            class_indices.append(max_index)
            
        except Exception as e:
            print(f"Error processing {row[C.DF_PATH_COL]}: {e}")
            scores.append(None)
            class_indices.append(None)

    # Add the scores back into the dataframe
    test_fold.loc[:, "test_score"] = scores
    test_fold.loc[:, "test_index"] = class_indices
    

    # Optional: save updated dataframe to CSV
    os.makedirs(os.path.join(CURRENT_SCRIPT_DIR, "results"), exist_ok=True)
    test_fold.to_csv(os.path.join(CURRENT_SCRIPT_DIR, "results", "inference_results.csv"), index=False)
    print("Inference completed. Results saved to inference_results.csv")


    
if __name__ == "__main__":
    main()