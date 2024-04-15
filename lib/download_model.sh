#!/bin/sh
wget https://github.com/brain-facens/SimSwap/releases/download/v1.0/arcface_model.zip
unzip arcface_model.zip
rm arcface_model.zip
wget https://github.com/brain-facens/SimSwap/releases/download/v1.0/checkpoints.zip
unzip checkpoints.zip
rm checkpoints.zip
wget https://github.com/brain-facens/SimSwap/releases/download/v1.0/insightface_func.zip
unzip insightface_func.zip
rm insightface_func.zip