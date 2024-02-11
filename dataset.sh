#!/bin/zsh
mkdir dataset

wget -P "dataset/" "https://data.caltech.edu/records/czs3p-5ss80/files/train.zip"
wget -P "dataset/" "https://data.caltech.edu/records/czs3p-5ss80/files/val.zip"
wget -P "dataset/" "https://data.caltech.edu/records/czs3p-5ss80/files/test.zip"

unzip -q "dataset/train.zip" -d "dataset/"
unzip -q "dataset/val.zip" -d "dataset/"
unzip -q "dataset/test.zip" -d "dataset/"