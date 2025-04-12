# finetune
me playing around with fine tuning.


## Setup

- Install pyenv
- Install pytorch && torchtune
- downloaded models from huggingface
    - `huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir /home/marek/models/meta-llama/Llama-2-7b-hf --token=$HUGGING_FACE_TOKEN`


### Learnings:
- Need to use the same tokenizer as the model was trained with
- Need the data to be formatted in the same way as the model was trained.

want to download a dataset? goodluck finding any info on it. I used this.
`huggingface-cli download yahma/alpaca-cleaned --local-dir /home/marek/datasets/alpaca-cleaned/ --token=$HUGGING_FACE_TOKEN --repo-type dataset`
