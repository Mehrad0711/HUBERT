#!/bin/bash

set -e
set -x

SRCDIR=`dirname $0`
CODEDIR=`dirname $SRCDIR`

WORKDIR=`mktemp -d $SRCDIR/hubert-tests-XXX`

function delete {
    rm -rf $1
}

# tests begin here
i=0

for encoder in  tpr_transformers tpr_lstm lstm no_enc ; do

    # train
    python3 $CODEDIR/run_model.py --task_name QQP --encoder $encoder --data_dir $CODEDIR/tests/sample_data/QQP/ --bert_model bert-base-uncased --do_train True --do_eval True --do_test False --output_dir $WORKDIR/trained_models/QQP --dSymbols 30 --dRoles 30 --nSymbols 50 --nRoles 35 --train_batch_size 2 --num_train_epochs 2 --delete_ok True

    # transfer 
    python3 run_model.py  --task_name MNLI --encoder $encoder --data_dir $CODEDIR/tests/sample_data/MNLI/ --bert_model bert-base-uncased --do_train True --do_eval True --load_ckpt $WORKDIR/trained_models/QQP/pytorch_model_best.bin --output_dir $WORKDIR/final_results/QQPtoMNLI --dSymbols 30 --dRoles 30 --nSymbols 50 --nRoles 35 --train_batch_size 2 --num_train_epochs 2 --load_bert_params False --load_role True --load_filler False --delete_ok True

    # predictions for HANS
    python3 run_model.py --no_cuda True --task_name HANS --encoder $encoder --data_dir $CODEDIR/tests/sample_data/HANS/ --bert_model bert-base-uncased  --do_test True --load_ckpt $WORKDIR/final_results/QQPtoMNLI/pytorch_model_best.bin --dSymbols 30 --dRoles 30 --nSymbols 50 --nRoles 35 --eval_batch_size 2 --output_dir $WORKDIR/final_results/QQPtoMNLI/hans/ --delete_ok True

    # check if result file exists
    if [ ! -s $WORKDIR/final_results/QQPtoMNLI/hans/test_predictions.txt ]; then
        echo "HANS prediction results not found!"
        exit 1
    fi

    i=$((i+1))

done

echo "All tests have passed!"

trap "delete $WORKDIR" TERM
