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

for encoder in tpr_transformers no_enc ; do

	# train on two tasks consecutively
	python3 run_model.py --no_cuda True --task_name SNLI --cont_task_names SST  --encoder $encoder --data_dir $CODEDIR/tests/sample_data/ --bert_model bert-base-uncased --do_train True --do_eval True --do_test False --output_dir $WORKDIR/continual_training/ --dSymbols 30 --dRoles 30 --nSymbols 50 --nRoles 35 --train_batch_size 1 --num_train_epochs 2 --num_bert_layers 1 --delete_ok True

    # train
    python3 run_model.py --no_cuda True --task_name QQP --encoder $encoder --data_dir $CODEDIR/tests/sample_data/ --bert_model bert-base-uncased --do_train True --do_eval True --do_test False --output_dir $WORKDIR/trained_models/ --train_batch_size 1 --num_train_epochs 2 --num_bert_layers 1 --delete_ok True

    # transfer 
    python3 run_model.py --no_cuda True  --task_name MNLI --encoder $encoder --data_dir $CODEDIR/tests/sample_data/ --bert_model bert-base-uncased --do_train True --do_eval True --load_ckpt $WORKDIR/trained_models/QQP/pytorch_model_best.bin --output_dir $WORKDIR/final_results/ --train_batch_size 1 --num_train_epochs 2 --load_bert_params False --load_role True --load_filler False --num_bert_layers 1 --delete_ok True

    # predictions for HANS
    python3 run_model.py --no_cuda True --task_name HANS --encoder $encoder --data_dir $CODEDIR/tests/sample_data/ --bert_model bert-base-uncased  --do_test True --load_ckpt $WORKDIR/final_results/MNLI/pytorch_model_best.bin  --eval_batch_size 1 --output_dir $WORKDIR/hans_results/ --num_bert_layers 1 --delete_ok True

    # check if result file exists
    if [ ! -s $WORKDIR/hans_results/HANS/test_predictions.txt ]; then
        echo "HANS prediction results not found!"
        exit 1
    fi

    i=$((i+1))

done

echo "All tests have passed!"

trap "delete $WORKDIR" TERM
