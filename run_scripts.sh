model_list=meta-llama/Llama-2-7b-hf,meta-llama/Llama-2-13b-hf
dataset_list=imdb,hate_speech_offensive,ag_news

for model in ${model_list//,/ }
do 
  for dataset in ${dataset_list//,/ }
  do
    python train.py -m $model -d $dataset
  done
done