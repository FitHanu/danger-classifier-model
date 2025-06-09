import site
import os
path = site.getsitepackages()[0]
files = os.listdir(path)

for file in files:
    print(file)