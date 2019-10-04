[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Travis-CI](https://api.travis-ci.com/Mehrad0711/HUBERT.svg?token=bGPLh8DGc3xJsVMrFqmZ&branch=master)](https://travis-ci.com/Mehrad0711/HUBERT/)

# HUBERT 

This repository contains the code implementation for HUBERT, as described in:

_HUBERT Untangles BERT to Improve Transfer across NLP Tasks_<br/>
Mehrad Moradshahi, Hamid Palangi, Monica S. Lam, Paul Smolensky, Jianfeng Gao<br/>
[Link](https://arxiv.org/...) 

## Quickstart

#### Setup Environment (Optional):
1. Intall python3.6: https://www.python.org/downloads/release/python-360/

2. Install required libraries </br>
   `> pip3 install -r requirements.txt`

<!--#### Use docker:-->
<!--1. Pull docker </br>-->
<!--   ```> docker pull allenlao/pytorch-mt-dnn:v0.21```-->

<!--2. Run docker </br>-->
<!--   ```> docker run -it --rm --runtime nvidia  allenlao/pytorch-mt-dnn:v0.21 bash``` </br>-->
<!--   Please refer to the following link if you first use docker: https://docs.docker.com/-->
### Dataset

1. Download GLUE, SNLI, and HANS: </br>
   ```> sh utils/download_data.sh $DATA_DIR``` </br>
    The files will be downloaded to $DATA_DIR (default path is './data'). Each folder within this directory contains the data splits for each task.

### Train and test model

<!-- 	```bash
	python3 ./data/HANS/evaluate_heur_output.py ./predictions/test_predictions.txt
	```
 -->

1. Training</br>
	```bash
	python3 run_model.py --task_name $task --encoder $encoder --data_dir ./data/ --bert_model bert-base-uncased --do_train True --do_eval False --do_test False --output_dir ./results/ --train_batch_size 256 --num_train_epochs 10
	```
	After every epoch, your model will be saved and evaluatd on dev set. If you wish to only keep the best model checkpoint (i.e. having the best accuracy on dev set) set `--save_best_only` to True. 
	Multi-GPU training is automatically on, so if you have more than 1 GPU, distributed training will be performed automatically. `--encoder` specifies the type of head used on top of BERT (e.g. LSTM, TPR_LSTM, TPR_Transformers)

2. Testing</br>
	```bash
	python3 run_model.py --no_cuda True --task_name $task --encoder $encoder --data_dir ./data/ --bert_model bert-base-uncased --do_train False --do_eval False --do_test True --load_ckpt ./results/$task/pytorch_model_best.bin --eval_batch_size 512 
	```
	setting `--no_cuda` to True will run the experiments on CPU. (thus, you can use a bigger batch_size for evaluation since CPU has more memory than GPU)


### Transfer Learning experiments
1. First, fine-tune HUBERT on the source-task you want to transfer the knowledge from. For example when your source task is MNLI:</br>
	```bash
	python3 run_model.py --task_name MNLI --data_dir ./data/ --bert_model bert-base-uncased --do_train True --dSymbols 30 --dRoles 30 --nSymbols 50 --nRoles 35 --num_train_epochs 10 --output_dir ./trained_models/ 
	```
you should get ~84% accuracy on MNLI matched dev set.

2. Second, load the fine-tuned model and initialize a second (only pre-trained) HUBERT model with the desired subset of paprameters from the former model. Then train, evalaute and test the model on the target task. For example when your target task is QQP and you only want to load roles:</br>
	```bash
	python3 run_model.py  --task_name QQP --data_dir ../data/ --bert_model bert-base-uncased --do_train True --do_eval True --load_ckpt ./trained_models/MNLI/pytorch_model_best.bin --output_dir ./final_results/ --num_train_epochs 10 --load_bert_params False --load_role True --load_filler False
	```

	You should get ~91% accuracy on QQP dev set.


3. To evaluate your models on HANS, first run this command to generate the predictions:</br>
	```bash
	python3 run_model.py --no_cuda True --task_name HANS --data_dir ./data/ --bert_model bert-base-uncased --do_test True --load_ckpt ./final_results/QQP/pytorch_model_best.bin --eval_batch_size 512 --output_dir ./predictions/
	```
	and then this command to produce the results broken down into different categories:</br>

	```bash
	python3 ./data/HANS/evaluate_heur_output.py ./predictions/HANS/test_predictions.txt
	```

## Continual Learning

Now you have the option to train your models on multiple datasets consecutively. Your source dataset shoud be specified with `--task_name` flag. Additional datasets are provided by using `--cont_task_names` flag and specifying the task names in comma separated format.</br> For source task, we train a HUBERT model for given number of epochs. We then choose the model with best accuracy on dev set, and use it to initialize a new model for the next task. Note that we use a fresh optimizer for each training task. For example if you want to train your model on SNLI and then fine-tune that on SST run:
```bash
python3 run_model.py --task_name SNLI --cont_task_names SST --data_dir ./data/ --bert_model bert-base-uncased --do_train True --do_eval True --do_test False --dSymbols 30 --dRoles 30 --nSymbols 50 --nRoles 35 --train_batch_size 256 --num_train_epochs 10  --output_dir ./continual_results/
```



### Notes on training
1. Gradient Accumulation </br>
  To make the training faster without much loss in accuracy you can use the `--gradient_accumulation_steps` argument. For example, if you set it to 3, the model will accumulate the gradients for 3 consecutive batches, and then perform one update step. 
2. FP16</br>
   We support FP16 training, however, note that our results are obtained using FP32.</br>
To use mixed-precision please install [apex](https://github.com/NVIDIA/apex) </br>, and set `--fp16` to True.

### TODOs

- [x] Add Travis for code testing
- [x] Support regression tasks (e.g. STS)
- [x] migrate from pytorch_pretrained_bert to transformers library
- [x] Option for continual training



## Acknowledgments
Our implementations are in PyTorch and based on the [HuggingFace](https://github.com/huggingface/pytorch-pretrained-BERT) and BERT’s [original codebase](https://github.com/google-research/bert).

### Citation
If you use this in your work, please cite HUBERT:

```
@inproceedings{,
    title = "HUBERT Untangles BERT to Improve Transfer across NLP Tasks",
}
```
