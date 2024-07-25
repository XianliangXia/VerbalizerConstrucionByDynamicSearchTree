# VerbalizerConstrucionByDynamicSearchTree Source Code
 This is the source code for our paper Construcion of prompt verbalizer based on dynamic search trees for text classification (DST).
 
## Install packages
- install Openprompt. You can get it at this link https://github.com/thunlp/OpenPrompt. There you will find out how to install it.
- install kpt. You can get it at this link https://github.com/thunlp/KnowledgeablePromptTuning. There you will find out how to install it.
- install transformers. You can get it at this link https://github.com/huggingface/transformers. There you will find out how to install it.

## Download datasets
- OpenPrompt provides the command to get the datasets. Use the following cmd.
~~~
cd OpenPrompt/datasets
bash download_text_classification.sh
~~~
## Download pre-trained language models
- Our approach can be adapted to a variety of models and tasks. So choose and download you PLM first!
- You can get the models from Transformers. An example code as follow.
~~~
from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, \
    RobertaForMaskedLM, AutoModelForMaskedLM, AutoTokenizer
def initialize_bertmodel():
    model_id = 'bert-base-uncased'
    bert = BertForMaskedLM.from_pretrained(model_id, local_files_only=True)

    bert.eval()
    tokenizer = BertTokenizer.from_pretrained(model_id)
    global mask
    mask = '[MASK]'
    return bert, tokenizer


def initialize_robertamodel():
    model_id = 'roberta-large'
    bert = RobertaForMaskedLM.from_pretrained(model_id, local_files_only=True)

    bert.eval()
    tokenizer = RobertaTokenizer.from_pretrained(model_id)
    return bert, tokenizer
~~~

## Run the code to construct a verbalizer
