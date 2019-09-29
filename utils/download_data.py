#!/usr/bin/env bash
############################################################## 
# This script is used to download dataset for HUBERT experiments
############################################################## 

DATA_DIR=$(pwd)/data
if [ ! -d ${DATA_DIR}  ]; then
  echo "Create a folder $DATA_DIR"
  mkdir ${DATA_DIR}
fi

## DOWNLOAD GLUE DATA
## Please refer glue-baseline install requirments or other issues.
git clone https://github.com/jsalt18-sentence-repl/jiant.git
cd jiant
python scripts/download_glue_data.py --data_dir $DATA_DIR --tasks all

cd ..
rm -rf jiant
#########################

## DOWNLOAD SNLI
cd $DATA_DIR
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip
mv snli_1.0 SNLI
cd $SNLI
for name in train dev test; do mv snli_1.0_"$name".txt "$name".tsv; done
cd ..
# remove zip files
rm *.zip

## Download HANS data and evaluation tool
git clone https://github.com/tommccoy1/hans.git -O hans_repo
mkdir HANS
cp heuristics_evaluation_set.txt evaluate_heur_output.py HANS/
rm -rf hans_repo
