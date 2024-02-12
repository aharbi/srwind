#!/bin/zsh
mkdir dataset

wget -P "dataset/" "https://data.caltech.edu/records/czs3p-5ss80/files/train.zip"
wget -P "dataset/" "https://data.caltech.edu/records/czs3p-5ss80/files/val.zip"
wget -P "dataset/" "https://data.caltech.edu/records/czs3p-5ss80/files/test.zip"

# Uncomment to use unzip instead of 7z (However, train.zip seems to corrupted)
#unzip -q "dataset/train.zip" -d "dataset/"
#unzip -q "dataset/val.zip" -d "dataset/"
#unzip -q "dataset/test.zip" -d "dataset/"

7z x "dataset/train.zip"
7z x "dataset/val.zip"
7z x "dataset/test.zip"