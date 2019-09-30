#!/usr/bin/env bash

set -x
set -e
############################################################## 
# This script is used to download dataset for HUBERT experiments
############################################################## 

DATA_DIR=$(pwd)/${1:-data}
if [ ! -d ${DATA_DIR}  ]; then
  echo "Create a folder $DATA_DIR"
  mkdir ${DATA_DIR}
fi

## DOWNLOAD GLUE DATA
git clone https://github.com/jsalt18-sentence-repl/jiant.git
cd jiant
python scripts/download_glue_data.py --data_dir $DATA_DIR --tasks all

cd ..
rm -rf jiant
#########################


## DOWNLOAD SNLI
## no need to download SNLI separately anymore since jiant does it
# cd $DATA_DIR
# wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
# unzip -u snli_1.0.zip
# mv snli_1.0 SNLI
# cd SNLI
# for name in train dev test; do mv snli_1.0_"$name".txt "$name".tsv; done
# rm -rf *.jsonl
# cd -
# # remove zip files
# rm *.zip

## Download HANS data and evaluation tool
cd $DATA_DIR
git clone https://github.com/tommccoy1/hans.git hans_repo
mkdir HANS
cp hans_repo/heuristics_evaluation_set.txt hans_repo/evaluate_heur_output.py HANS/
rm -rf hans_repo

# rename to match our code
cd $DATA_DIR
mv SST-2 SST
mv STS-B STS
