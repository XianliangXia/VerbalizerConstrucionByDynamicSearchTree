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

def initialize_robertamodel():
    model_id = 'roberta-large'
    bert = RobertaForMaskedLM.from_pretrained(model_id, local_files_only=True)

    bert.eval()
    tokenizer = RobertaTokenizer.from_pretrained(model_id)
    return bert, tokenizer
~~~

## Run the code to construct a verbalizer

### PDS
- First, you need to find the positive dimensions (PDs), the code in the file positive dimensions selection.py.
- Note that you can run it in Jupyter Notebook to save time for loading models.
- There you can give different k values until you get a satisfactory result.
  
### Use DST to construct a verbalizer
- When you get the PDs, use them in the next step.
- They will be working on the rectified cosine similarity to measure the relationship between different words in a specified feature.
- The code in the DST.py
- put your PDs here.
~~~~
meaning_dim = [7, 83, 94, 113, 170, 195, 289, 296, 347, 350, 351, 398, 401,
             447, 491, 527, 532, 554, 570, 621, 669, 674, 679, 705, 743, 768,
             795, 827, 862, 906, 950, 952]
~~~~

### Use the verbalizer for tasks
- When get the verbalizer, use it in KPT Integration.py
- You can run it by run_classificaion.sh. Note that change the dir to the right path.
~~~~
bash run_classificaion.sh
~~~~
