"""
============================
# -*- coding: utf-8 -*-
# @Author  : xianlaing.xia
# @FileName: prompt_tuning.py
# @Software: PyCharm
# @Des     : 
===========================
"""
import json

import torch
from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, \
    RobertaForMaskedLM, AutoModelForMaskedLM, AutoTokenizer

import matplotlib.pyplot as plt
import os
from pyecharts import options as opts
from pyecharts.charts import Tree, Page
import datetime
import pandas as pd
from tqdm import tqdm
from tqdm.auto import trange
import nltk
from nltk.corpus import stopwords
import numpy as np
import seaborn

from torchnlp.datasets import imdb_dataset
from datasets import load_dataset
import random

# 1 dataset
dataset_name = {
    1: "smss",
    2: "IMDB",
    3: "amazon",
    4: "mr",
    5: "yelp",

}
device = torch.device('cuda')

mask = '<mask>'

global bert, tokenizer, plm


# pakg 0-positive 1-negative
def get_data(dataset):
    if dataset == dataset_name[1]:
        return get_SMSS()
    elif dataset == dataset_name[2]:
        return get_IMDB()
    elif dataset == dataset_name[3]:
        return get_amazon()
    elif dataset == dataset_name[4]:
        return get_mr()
    elif dataset == dataset_name[5]:
        return get_yelp()


def get_IMDB():
    test = imdb_dataset('./dataset', test=True)
    random.seed = 666
    print(len(test))
    # pakg = random.sample(pakg, k=1000)
    labels = []
    msgs = []
    for item in test:
        labels.append(0 if item.get('sentiment') == 'pos' else 1)
        msgs.append(item.get('text'))
    print(labels.count(0), labels.count(1))
    return labels, msgs


def get_amazon():
    data = load_dataset('amazon_polarity', split='pakg')
    random.seed(888)
    print(len(data))
    data = random.sample(list(data), k=10000)
    labels = []
    msgs = []
    for item in data:
        labels.append(0 if item.get('label') == 1 else 1)
        msgs.append(item.get('title') + item.get('content'))
    print(labels.count(0), labels.count(1))
    return labels, msgs


def get_mr():
    # 0n 1p
    data = load_dataset('mattymchen/mr', split='pakg')
    random.seed(888)
    len_data = len(data)
    print(len_data)
    labels = []
    msgs = []
    for item in data:
        labels.append(0 if item.get('label') == 1 else 1)
        msgs.append(item.get('text'))
    print(labels.count(0), labels.count(1))
    return labels, msgs


def get_yelp():
    # 0n 1p
    data = load_dataset('yelp_polarity', split='pakg')
    random.seed(888)
    len_data = len(data)
    print(len_data)
    labels = []
    msgs = []
    for item in data:
        labels.append(0 if item.get('label') == 1 else 1)
        msgs.append(item.get('text'))
    print(labels.count(0), labels.count(1))
    return labels, msgs


class MyTree:

    def __init__(self, node):
        self.layer = {1: [[node]]}

    def get_elements(self):
        temp = []
        for _, v in self.layer.items():
            reduction = sum(v, [])
            temp.extend(reduction)
        return temp



def initialize_robertamodel():
    model_id = 'roberta-large'
    bert = RobertaForMaskedLM.from_pretrained(model_id, local_files_only=True)

    bert.eval()
    tokenizer = RobertaTokenizer.from_pretrained(model_id)
    return bert, tokenizer


def construct_data(prompt_begin, seq, prompt_end):
    return {'b': prompt_begin.strip(), 'c': seq.strip(), 'e': prompt_end.strip()}


def seq2index(data, tokenizer):
    bg = data['b']
    ed = data['e']
    ct = data['c']
    max_ct_len = 0

    token = tokenizer(text=' '.join(data.values()).strip(), truncation=False, return_tensors='pt')
    if token['input_ids'].shape[1] > 512:
        if len(bg) != 0:
            token_b = tokenizer(text=bg, return_tensors='pt')
            max_ct_len += token_b['input_ids'].shape[1] - 2
        if len(ed) != 0:
            token_e = tokenizer(text=ed, return_tensors='pt')
            max_ct_len += token_e['input_ids'].shape[1] - 2
        token_c = tokenizer(text=ct, truncation=True, max_length=512 - max_ct_len, return_tensors='pt')
        if len(bg) != 0:
            inps = torch.concat([token_b['input_ids'][:, :-1], token_c['input_ids'][:, 1:]], dim=1)
            if len(ed) != 0:
                inps = torch.concat([inps, token_e['input_ids']], dim=1)
        else:
            inps = torch.concat([token_c['input_ids'][:, :-1], token_e['input_ids'][:, 1:]], dim=1)
        token['input_ids'] = inps
        token['attention_mask'] = token['attention_mask'][:, :512]
        if mask == '[MASK]':
            token['token_type_ids'] = token['token_type_ids'][:, :512]

    # token = tokenizer.encode_plus(text=data, truncation=True, return_tensors='pt')

    token_index = token['input_ids']
    token_text = tokenizer.convert_ids_to_tokens(token_index.flatten().tolist())

    # bert
    pre_index = token_text.index(mask)
    return token, pre_index


def get_result(token_pkg):
    res = bert(**token_pkg)
    return res


def show_topk_result(res, k):
    predicted_index = torch.topk(res, k)
    for i in predicted_index.indices:
        predicted_token = tokenizer.convert_ids_to_tokens([i])[0]
        print(predicted_token)


# which words we should store as the node

# Cosine similarity
def get_CorVaerlizerTree_by_CosineSimilarity(aspect1, aspect2, tokenizer, word_embedding):
    aspect1_list = [aspect1]
    aspect2_list = [aspect2]
    cosine_similarity_list = []
    cosine_threshold = 0.42
    depth_threshold = 3

    a = word_embedding(torch.tensor(tokenizer.convert_tokens_to_ids('good')))
    b = word_embedding(torch.tensor(tokenizer.convert_tokens_to_ids('bad')))
    test_cosine = get_cosine_similarity(a, b)
    print(test_cosine)

    root = tokenizer.convert_tokens_to_ids(aspect1)
    root2vec = word_embedding(torch.tensor(root))

    for k, v in tokenizer.vocab.items():
        if len(k) >= 2 and k.isalpha() and (k not in aspect1_list):
            v2vec = word_embedding(torch.tensor(v))
            cosine_similarity = get_cosine_similarity(root2vec, v2vec)
            cosine_similarity_list.append(cosine_similarity.item())
            if cosine_similarity >= cosine_threshold:
                aspect1_list.append(k)
    print(cosine_similarity_list)
    # draw
    plt.hist(cosine_similarity_list, bins=200)
    plt.xlabel("range")
    plt.ylabel("num")
    plt.show()

    return {"aspect1": aspect1_list, "aspect2": aspect2_list}


# CosineSimilarity KNN
def get_CorVaerlizerTree_by_CosineSimilarityKNN(aspect1, aspect2, tokenizer, word_embedding):
    layer = 3
    k_num = 3

    w1_tree = generate_tree(aspect1, tokenizer, word_embedding, k_num, layer)
    w2_tree = generate_tree(aspect2, tokenizer, word_embedding, k_num, layer)
    print("o1:", w1_tree.get_elements())
    print("o2:", w2_tree.get_elements())

    w1_tree = [tokenizer.convert_tokens_to_ids(elmt) for elmt in w1_tree.get_elements()]
    w2_tree = [tokenizer.convert_tokens_to_ids(elmt) for elmt in w2_tree.get_elements()]
    print({"o1": w1_tree, "o2": w2_tree})
    return {"o1": w1_tree, "o2": w2_tree}


def get_CorVaerlizerTree_by_CosineSimilarityKNN_list(label_list, tokenizer, word_embedding):

    layer = 3
    k_num = 2

    extend_label = []
    for label_idx, label_list_sub in enumerate(label_list):
        temp_list = []
        bar = tqdm(label_list_sub)
        for root in bar:
            bar.set_description_str(f'label index {label_idx:02d}')
            dst = generate_tree(root, tokenizer, word_embedding, k_num, layer)
            # remove duplicate
            for word in dst.get_elements():
                if word not in temp_list:
                    temp_list.append(word)
        extend_label.append(temp_list)
    return extend_label  # [[],]


def get_distance(vec1, vec2):
    return torch.norm(vec1 - vec2)


def get_inner(root, vec):
    return torch.inner(root, vec)


def get_cosine_similarity(vec1, vec2):
    return torch.inner(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))


def get_cosine_distance(vec1, vec2):
    return 1 - torch.inner(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))


def generate_tree(root, tokenizer, word_embedding, k_num, layer):
    res = MyTree(root)
    count = 1
    goal_word = nltk.pos_tag([root])[0][1]

    while count <= layer - 1:
        temp_roots = res.layer[count]
        temp_nodes = []
        exist_words = res.get_elements()
        for root_item in sum(temp_roots, []):
            cosine_similarity_list = []
            k_with_vector_dict = {}

            root_id = tokenizer.convert_tokens_to_ids(root_item)
            root2vec = word_embedding(torch.tensor(root_id).to(device))
            for k, v in tokenizer.get_vocab().items():  # k-word v-index
                if len(k) >= 3 and (ord(k[0]) in range(97, 123)) and (k not in exist_words):
                    if nltk.pos_tag([k])[0][1] == goal_word:
                        v2vec = word_embedding(torch.tensor(v).to(device))

                        meaning_dim = [7, 83, 94, 113, 170, 195, 289, 296, 347, 350, 351, 398, 401,
                                       447, 491, 527, 532, 554, 570, 621, 669, 674, 679, 705, 743, 768,
                                       795, 827, 862, 906, 950, 952]


                        root2vec_meaning = [root2vec[i] for i in meaning_dim]
                        v2vec_meaning = [v2vec[i] for i in meaning_dim]
                        root2vec_meaning = torch.tensor(root2vec_meaning)
                        v2vec_meaning = torch.tensor(v2vec_meaning)

                        cosine_similarity = get_cosine_similarity(root2vec_meaning, v2vec_meaning)
                        # if cosine_similarity >= 0.66:
                        cosine_similarity_list.append(cosine_similarity.item())
                        k_with_vector_dict[k] = cosine_similarity.item()
            knn_res = torch.topk(torch.tensor(list(k_with_vector_dict.values())), k_num)


            node_element = []
            for i in knn_res.indices:
                node_element.append(list(k_with_vector_dict.keys())[i])
            exist_words.extend(node_element)
            temp_nodes.append(node_element)
        count += 1
        res.layer[count] = temp_nodes
    # draw_tree(res)
    return res


def draw_tree(my_tree: MyTree):
    roots = sum(my_tree.layer[1], [])
    depth = len(my_tree.layer)
    operate_level = 1
    res_data = [{"name": "", "children": []}]
    children_list_up_level = [res_data[0]["children"]]

    while operate_level <= depth:
        level_elements = my_tree.layer[operate_level]
        temp_children_list_up_level = []
        for place, nodes in enumerate(level_elements):  # [l,m] 0
            for node in nodes:  # a
                temp = {"name": node, "children": []}
                temp_children_list_up_level.append(temp["children"])
                children_list_up_level[place].append(temp)
        children_list_up_level = temp_children_list_up_level
        operate_level += 1

    tree = (
        Tree()
        .add("", res_data,
             label_opts=opts.LabelOpts(
                 position="top",
                 horizontal_align="right",
                 vertical_align="middle",
                 rotate=0,
             ), )
        .set_global_opts(title_opts=opts.TitleOpts(title=f"show_words_tree:{roots[0]}"))
    )
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%Hh-%Mm-%Ss')
    prefix_path = "./trees"
    if not os.path.exists(prefix_path):
        os.mkdir(prefix_path)
    root_plain = str(roots).replace("'", '')
    tree.render(f"{prefix_path}/{root_plain}_tree_{timestamp}.html")


def verbalizer(cor_words, res, real_label):
    t1, t2 = cor_words["o1"], cor_words['o2']
    o1 = count_probability(t1, res)
    o2 = count_probability(t2, res)
    pre = 1
    if o1 > o2:
        pre = 0
    if pre == real_label:
        return 1
    return 0


def count_probability(t, res):
    total = 0
    for idx in t:
        total += res[idx]
    return total


def get_IW_manual():
    initial_aspect1 = "positive"
    initial_aspect2 = "negative"
    return initial_aspect1, initial_aspect2


def get_IW_auto(pb, pe):
    data = construct_data(prompt_begin=pb, seq="", prompt_end=pe)
    token, mask_idx = seq2index(data, tokenizer)
    # res
    token = token.to(device)
    result = get_result(token)
    result = result[0][0][mask_idx]
    initial_aspect1 = "normal"
    initial_aspect2 = "advertisement"
    return initial_aspect1, initial_aspect2


def choice_prompts(dataset, idx):
    prompt_begin, prompt_end = '', ''
    if dataset == dataset_name[1]:
        if idx == 1:
            prompt_begin = f"Just {mask} ."
            prompt_end = ""
        elif idx == 2:
            prompt_begin = f"It was {mask} ."
            prompt_end = ""
        elif idx == 3:
            prompt_begin = ""
            prompt_end = f"All in all, it was {mask} ."
        elif idx == 4:
            prompt_begin = ""
            prompt_end = f'In summary, the email was {mask} ".'
    if dataset == dataset_name[2]:
        if idx == 1:
            prompt_begin = f"Just {mask} !"
            prompt_end = ""
        elif idx == 2:
            prompt_begin = f"It was {mask} ."
            prompt_end = ""
        elif idx == 3:
            prompt_begin = ""
            prompt_end = f"All in all, it was {mask} ."
        elif idx == 4:
            prompt_begin = ""
            prompt_end = f'In summary, the film was {mask} .'
    if dataset in [dataset_name[3], dataset_name[4], dataset_name[5]]:
        if idx == 1:
            prompt_begin = f"Just {mask} !"
            prompt_end = ""
        elif idx == 2:
            prompt_begin = f"It was {mask} ."
            prompt_end = ""
        elif idx == 3:
            prompt_begin = ""
            prompt_end = f"All in all, it was {mask} ."
        elif idx == 4:
            prompt_begin = ""
            prompt_end = f'In summary, it was {mask} ".'

    print(prompt_begin, prompt_end)
    return prompt_begin, prompt_end


def main():
    dats_idx = 4
    print(dataset_name[dats_idx])
    va_labels, va_msgs = get_data(dataset_name[dats_idx])

    # 2 load model, bert is model, tokenizer is used for changing idx and word
    global bert, tokenizer
    bert, tokenizer = initialize_robertamodel()
    bert.to(device)

    # 3 design prompt
    prompt_begin, prompt_end = choice_prompts(dataset_name[dats_idx], 4)

    # 4 CorVerbalizer
    initial_aspect1, initial_aspect2 = get_IW_manual()
    approaches = {5: get_CorVaerlizerTree_by_CosineSimilarityKNN}

    selected_approach = 5

    embedding = bert.get_input_embeddings()

    cor_words = choice_saved_corwords(dataset_name[dats_idx], 21)  # 6


    print(cor_words)
    print(tokenizer.convert_ids_to_tokens(cor_words['o1']), tokenizer.convert_ids_to_tokens(cor_words['o2']))

    right_num = 0
    for c, i in enumerate(tqdm(va_msgs)):
        msg = i

        data = construct_data(prompt_begin=prompt_begin, seq=msg, prompt_end=prompt_end)

        token, mask_idx = seq2index(data, tokenizer)

        token = token.to(device)
        result = get_result(token)
        result = result[0][0][mask_idx]
        right_num += verbalizer(cor_words, result, va_labels[c])
    print("right_num:", right_num)


def test_():
    bert, tokenizer = initialize_robertamodel()
    print(tokenizer.get_vocab())

def test_DStree():
    global bert, tokenizer
    bert, tokenizer = initialize_bertmodel()
    bert.to(device)
    embedding = bert.get_input_embeddings()

    initial_aspect1, initial_aspect2 = get_IW_manual()
    get_CorVaerlizerTree_by_CosineSimilarityKNN(initial_aspect1, initial_aspect2, tokenizer, embedding)


def get_kpt_words(data_set_name):
    label_list = []
    if data_set_name != 'yahoo_answers_topics':
        path = '../OpenPrompt/scripts/TextClassification/'+data_set_name+f'/knowledgeable_verbalizer.txt'
        with open(path, 'r') as vfile:
            words_list = vfile.readlines()
            words_list = [line.strip().split(',') for line in words_list]
            for i in words_list:
                label_list.append(i)
    else:
        path = '../OpenPrompt/scripts/TextClassification/' + data_set_name + f'/knowledgeable_verbalizer.json'
        with open(path, 'r') as vfile:
            words_dict = json.load(vfile)
            for i in words_dict.values():
                label_list.append(i)

    return label_list


def get_extend_word(idx):
    global bert, tokenizer
    data_set_name = ['amazon', 'imdb', 'dbpedia',  'agnews', 'yahoo_answers_topics']
    bert, tokenizer = initialize_robertamodel()
    bert.to(device)
    data = data_set_name[idx]
    embedding = bert.get_input_embeddings()
    initial_list = get_kpt_words(data)[4:6]
    # todo 扩展
    label_list = get_CorVaerlizerTree_by_CosineSimilarityKNN_list(initial_list, tokenizer, embedding)
    print(label_list)
    with open(f'./dst_kpt_ver/{data}_0_1.txt', 'w+', encoding='utf-8') as resf:
        for sub_list in label_list:
            line = str(sub_list).replace('[', '').replace(']', '').replace("'", '').replace(' ', '')
            resf.write(line + ' \n')
    # verbalizer_format(data)

def verbalizer_format(name):
    data = name
    with open(f'./dst_kpt_ver/{data}_format.txt', 'w+') as new_resf:
        with open(f'./dst_kpt_ver/{data}.txt', 'r') as resf:
            cont = resf.readlines()
            for i in cont:
                data = i.replace('[', '').replace(']', '').replace("'", '').replace("\n", '').replace(' ', '')
                new_resf.write(data + '\n')


if __name__ == '__main__':
    get_extend_word(4)
