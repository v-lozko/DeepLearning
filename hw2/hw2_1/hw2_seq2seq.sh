#!/bin/bash

set -e

# Input arguments
DATA_DIR=$1
OUTPUT_FILE=$2


if [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_FILE" ]; then
  echo "Usage: ./hw2_seq2seq.sh <data_directory> <output_file>"
  exit 1
fi

DOWNLOAD_URL="https://www.dropbox.com/scl/fi/tnacteccaxpbt7a8f8ls6/seq2seq_model.pth?rlkey=l7iwr0jyzj2y2lb6gopgkjy5w&st=uud7ikhb&dl=0"
wget $DOWNLOAD_URL

python3 model_seq2seq.py "$DATA_DIR" "$OUTPUT_FILE"

echo "Output saved to $OUTPUT_FILE."

python3 bleu_eval.py "$OUTPUT_FILE"
