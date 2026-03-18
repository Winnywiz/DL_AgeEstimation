#!/bin/bash

NAME=fgnet-dataset
curl -L -o ./${NAME}.zip \
  https://www.kaggle.com/api/v1/datasets/download/aiolapo/fgnet-dataset
unzip -q ./${NAME}.zip -d ./dataset
rm -f ./${NAME}.zip