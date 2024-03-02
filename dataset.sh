#!/bin/zsh
mkdir dataset

# Uncomment to download the raw dataset
# wget -P "dataset/" "https://data.caltech.edu/records/czs3p-5ss80/files/train.zip"
# wget -P "dataset/" "https://data.caltech.edu/records/czs3p-5ss80/files/val.zip"
# wget -P "dataset/" "https://data.caltech.edu/records/czs3p-5ss80/files/test.zip"

# Uncomment to use unzip instead of 7z (However, train.zip seems to be corrupted)
# unzip -q "dataset/train.zip" -d "dataset/"
# unzip -q "dataset/val.zip" -d "dataset/"
# unzip -q "dataset/test.zip" -d "dataset/"

# Uncomment to use 7z instead of unzip
# 7z x "dataset/train.zip" -o./dataset/
# 7z x "dataset/val.zip" -o./dataset/
# 7z x "dataset/test.zip" -o./dataset/

# Download the processed dataset
gdown --id "1erKwFcJNoPMFRRE20bj4cDFZtWI0XbCn" -O "dataset/"
gdown --id "1ZZg-fxkzauLkfn9nNSRrtYryrwCyKjgS" -O "dataset/"