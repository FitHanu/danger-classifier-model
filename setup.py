import os
import subprocess
import importlib
import json
import sys
import argparse


from logging_cfg import get_logger
l = get_logger(__name__)

from constants import PY_PROJECT_ROOT, SITE_PKG_PATH

#Module registry
MODULE_REG = [
    os.path.join(SITE_PKG_PATH, PY_PROJECT_ROOT),
    os.path.join(SITE_PKG_PATH, PY_PROJECT_ROOT, "utils"),
    os.path.join(SITE_PKG_PATH, PY_PROJECT_ROOT, "ds"),
    os.path.join(SITE_PKG_PATH, PY_PROJECT_ROOT, "partition"),
    # os.path.join(SITE_PKG_PATH, PY_PROJECT_ROOT, "ext_models"),
]


def check_conda_installed():
    """Check if Conda or Miniconda is installed on the system."""
    try:
        # Try running 'conda --version'
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True, check=True)
        conda_version = result.stdout.strip()
        
        # Check if it's Miniconda or full Anaconda
        conda_info = subprocess.run(["conda", "info", "--json"], capture_output=True, text=True, check=True)
        if "conda" in conda_info.stdout.lower():
            l.info(f"conda detected: {conda_version}")
    
    except FileNotFoundError:
        l.error("Conda is NOT installed.")
        raise FileNotFoundError("Conda is NOT installed.")
        
    except subprocess.CalledProcessError:
        l.error("Conda is installed but not working properly.")
        raise subprocess.CalledProcessError("Conda is installed but not working properly.")

def install_requirements(strategy = "conda"):
    package_strategy = ["pip", "conda"]
    if strategy not in package_strategy:
        l.error(f"Strategy must be one of {package_strategy}")
        raise ValueError(f"Strategy must be one of {package_strategy}")
    if strategy == "pip":
        req_file = os.path.join(PY_PROJECT_ROOT, "requirements.txt")
        # pip install -r requirements.txt
        args = ["pip", "install", "-r", req_file]
        try:
            subprocess.call(args)
            l.info(f"Successfully installed dependencies from requirements.txt")
        except Exception as e:
            l.error(f"Failed to install requirements: {e}")
    else:
        check_conda_installed()
        req_file = os.path.join(PY_PROJECT_ROOT, "environment.yml")
        # conda env create -f environment.yml
        args = ["conda", "env", "create", "-f", req_file]
        try:
            subprocess.call(args)
            l.info(f"Successfully installed dependencies from environment.yml")
        except Exception as e:
            l.error(f"Failed to install requirements: {e}")

def append_project_path():
    """
    Python modules type shit
    """
    pth_file = SITE_PKG_PATH + os.path.sep + "sra.pth"
    with open(pth_file, "w") as f:
        for module in MODULE_REG:
            f.write(module + "\n")

def validate_packages(package_names):
    """Checks if required packages are available after installation."""
    for package in package_names:
        try:
            importlib.import_module(package)
            l.info(f"✅ {package} is available.")
        except ImportError:
            l.warning(f"❌ {package} is missing or failed to install.")
            

def init_split_ds_config(overwrite = False):
    """
    Will init partition configs to config.json
    
    """
    from constants import CONFIG_JSON
    pt_key = "partition"
    with open(CONFIG_JSON, "r") as f:
        config = json.load(f)
        if pt_key in config and overwrite == False:
            l.warning(f"Overwrite is set to off but \"{pt_key}\" key is already exists. Skipping ...")
            pass
        else:
            # Default partition config
            config = {pt_key: {
                "train": 0.8,
                "dev": 0.1,
                "test": 0.1
            }, **config}
    
    with open(CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)
    


def get_args():
    parser = argparse.ArgumentParser(description="Setup script for the project.")
    parser.add_argument(
        "--env_strategy",
        choices=["pip", "conda"],
        default="pip",
        help="Environment setup strategy. Choose between 'pip' or 'conda'. Default is 'pip'"
    )
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    env_strategy = args.env_strategy
    l.info("Arguments:")
    l.info(f"env_strategy: {env_strategy}")
    
    l.info("Setting up project ...")
    steps = [
        [f"Installing requirements", install_requirements, env_strategy],
        [f"Appending project paths to {SITE_PKG_PATH}", append_project_path],
        [f"Initialize default paritioning config", init_split_ds_config],
        [f"Validating packages ...", validate_packages, [
            "ds.dataset", "utils.json_utils"]],
    ]

    for step in steps:
        try:
            l.info(f"Running: {step[0]}")
            if len(step) > 2:
                step[1](step[2])
            else:
                step[1]()
        except Exception as e:
            l.error(f"Failed to run: {step[0]}")
            l.error(e)
            sys.exit(1)

if __name__ == "__main__":
    main()