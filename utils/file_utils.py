from json_utils import get_default_class_mapping
import subprocess
import zipfile
import uuid
from platformdirs import user_cache_dir
import os

def is_path(path: str) -> bool:
    """
    Validate if the given path is a valid directory path.
    """
    return os.path.isdir(path)

def init_class_folds(path: str):
    """
    Initialize class folders to `path`
    """
    if not is_path(path):
        raise ValueError(f"{path} is not a valid directory path.")

    default_class = get_default_class_mapping()
    for k, _ in default_class.items():
        os.makedirs(os.path.join(path, str(k)), exist_ok=True)
        

def get_filename_without_extension(file: str) -> str:
    """
    Get the filename without extension from the given file path.
    """
    return os.path.splitext(os.path.basename(file))[0]

def remove_extensions(file_name):
    names = file_name.split(".")
    return names[0]

def get_filename_from_path(file_path: str) -> str:
    """
    Get the filename from the given file path.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return os.path.basename(file_path)

def extract(file_path, output_path=".") -> None:
    print(f"Extract {file_path} to {output_path}")
    output_file_path = os.path.join(output_path, remove_extensions(file_path))
    os.makedirs(output_path, exist_ok=True)
    if file_path.endswith(".zip"):
        print("Handling zip file ...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(output_file_path)
    elif file_path.endswith(".tar.gz"):
        print("Handling .tar.gz file ...")
        subprocess.run(["tar", "-xzvf", file_path, "-C", output_file_path])
    else:
        print(f"Unknown file format {file_path}")


def get_file_name_from_url(url) -> str: 
    return url.split('/')[-1]

def download(url, file_path=".", file_name=None) -> None:
    if file_name is None:
        tmp_name = get_file_name_from_url(url)
        if (tmp_name == ""):
            print("Cannot get file name from URL")
            file_name = str(uuid.uuid4())
        else:
            file_name = get_file_name_from_url(url)
    local_filepath = os.path.join(file_path, file_name)
    print(f"Download {file_name} to {file_path}")
    subprocess.run(["wget", url, "-O", local_filepath])
    print(f"Done downloading, file name: {local_filepath}")

    return local_filepath


def clean_user_cache_dir():
    """
    Clean user cache dir \n
    In this context
    remove all cached downloaded dataset contents 
    """
    cache_dir = user_cache_dir()
    try:
        subprocess.run(["rm", "-rf", cache_dir])
        return cache_dir
    except Exception as e:
        raise e

def clean_dataset_dir():
    from constants import FILTERED_DATASET_PATH
    rm_path = os.path.join(FILTERED_DATASET_PATH, "*")
    try:
        subprocess.run(["rm", "-rf", rm_path])
    except Exception as e:
        raise e
    

def init_folds(final_path: str, fold_num: int) -> None:
    """
    Initialize fold directories to final dataset path
    """
    for i in range(fold_num):
        f_path = os.path.join(final_path, str(i))
        os.makedirs(f_path, exist_ok=True)
        



def main():
    clean_user_cache_dir()
    clean_dataset_dir()


if __name__ == "__main__":
    main()