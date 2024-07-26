PYTHONPATH=python
BASEPATH="../"
DATASET=agnews #agnews dbpedia imdb amazon yahoo
#TEMPLATEID=0 # 1 2 3
SEED=123 # 145 146 147 148
SHOT=0 # 0 1 10 20
CUT='--nocut'
#VERBALIZER=dst_kpt #kpt
CALIBRATION="--calibration" # "" --calibration
FILTER=tfidf_filter # none tfidf_filter
MODEL_NAME_OR_PATH="../plm_cache/roberta-large/snapshots/716877d372b884cad6d419d828bac6c85b3b18d9/"
RESULTPATH="results_zeroshot.txt"
OPENPROMPTPATH="../OpenPrompt"

cd $BASEPATH

CUDA_VISIBLE_DEVICES=0 $PYTHONPATH zeroshot.py \
--model_name_or_path $MODEL_NAME_OR_PATH \
--result_file $RESULTPATH \
--openprompt_path $OPENPROMPTPATH \
--dataset $DATASET \
--template_id 0 \
--seed $SEED \
--verbalizer kpt $CALIBRATION \
--filter $FILTER $CUT