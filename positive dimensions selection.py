#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
import seaborn
import pandas as pd


# In[77]:


import matplotlib.pyplot as plt


# In[154]:


plt.rcParams['axes.unicode_minus'] =False


# In[2]:


global bert, tokenizer


# In[3]:


def initialize_robertamodel():
    model_id = 'roberta-large'
    bert = RobertaForMaskedLM.from_pretrained(model_id)

    bert.eval()
    tokenizer = RobertaTokenizer.from_pretrained(model_id)
    return bert, tokenizer


# In[291]:


def initialize_xlmrobertamodel():
    model_id = 'xlm-roberta-base'
    bert = AutoModelForMaskedLM.from_pretrained(model_id)

    bert.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return bert, tokenizer


# In[320]:


def initialize_bertmodel():
    model_id = 'bert-base-uncased'
    bert = AutoModelForMaskedLM.from_pretrained(model_id)

    bert.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return bert, tokenizer


# In[4]:


def get_cosine_similarity(vec1, vec2):
    return torch.inner(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))


# In[ ]:


global bert, tokenizer
bert, tokenizer = initialize_robertamodel()



# In[10]:


def vec_dim_analysis():
    embedding = bert.get_input_embeddings()
    w1 = 'negative'
    w2 = 'bad'
    w3 = 'positive'
    # w1 = 'service'
    # w2 = 'bad'
    # w3 = 'normal'
    w1_vec = embedding(torch.tensor(tokenizer.convert_tokens_to_ids(w1)))
    seaborn.lineplot(w1_vec.detach())
    w2_vec = embedding(torch.tensor(tokenizer.convert_tokens_to_ids(w2)))
    w3_vec = embedding(torch.tensor(tokenizer.convert_tokens_to_ids(w3)))
    print(f'{w1}--{w2}:', get_cosine_similarity( w1_vec, w2_vec))  # tensor(0.6438, grad_fn=<DivBackward0>)
    print(f'{w1}--{w3}:', get_cosine_similarity(w1_vec, w3_vec))  # tensor(0.6125, grad_fn=<DivBackward0>)
    print(f'{w2}--{w3}:', get_cosine_similarity(w2_vec, w3_vec))
    res_o = []
    res_c = []
    for i in range(len(w1_vec)):
        res_c.append(np.linalg.norm([w1_vec[i].item()-w2_vec[i].item()]))
        res_o.append(np.linalg.norm([w1_vec[i].item()-w3_vec[i].item()]))
    return res_c, res_o


# In[9]:


def vic_distance():
    embedding = bert.get_input_embeddings()
    w1 = 'negative'
    w2 = 'bad'
    w3 = 'positive'
    w1_vec = embedding(torch.tensor(tokenizer.convert_tokens_to_ids(w1)))
    w2_vec = embedding(torch.tensor(tokenizer.convert_tokens_to_ids(w2)))
    w3_vec = embedding(torch.tensor(tokenizer.convert_tokens_to_ids(w3)))
    print(f'{w1}--{w2}:', get_cosine_similarity( w1_vec, w2_vec))  # tensor(0.6438, grad_fn=<DivBackward0>)
    print(f'{w1}--{w3}:', get_cosine_similarity(w1_vec, w3_vec))  # tensor(0.6125, grad_fn=<DivBackward0>)
    print(f'{w2}--{w3}:', get_cosine_similarity(w2_vec, w3_vec)) 
vic_distance()


# In[11]:


if __name__ == '__main__':
    rec, reo = vec_dim_analysis()


# In[135]:


data_1 = {
    "维度下标":range(len(reo)),
    '距离值':reo
}
data_1 = pd.DataFrame(data_1)

minimum_values = data_1['距离值'].sort_values(ascending=False).head(32)
print(minimum_values.values.tolist())


# In[142]:


seaborn.scatterplot(x="维度下标",y="距离值",data=data_1, marker='X',color="#ff7f0e").set(title="wb与wz各维度距离")
plt.scatter(x=minimum_values.index.tolist(),y=minimum_values.values.tolist(), marker='X',color="#20c99b")
plt.savefig("wbwz.png",dpi=400)


# In[21]:


print(reo)


# In[ ]:


# In[13]:


seaborn.scatterplot(rec)


# In[80]:


data_2 = {
    "维度下标":range(len(rec)),
    '距离值':rec
}
data_2 = pd.DataFrame(data_2)
print(data_2)


# In[81]:


seaborn.scatterplot(x="维度下标",y="距离值",data=data_2).set(title="wb与wc各维度距离")
plt.savefig("wbwc.png",dpi=400)



# In[177]:


data_o = {
    
          "维度下标":range(len(rec)),
          '词义相近':rec,
            '词义相反':reo
}
data_o = pd.DataFrame(data_o)
print(data_o)


# In[178]:


from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
seaborn.scatterplot(data=data_o)


# In[179]:


seaborn.jointplot(data_o)


# In[186]:


seaborn.displot(data=data_o['词义相反'], kde=True)


# In[303]:


np.argpartition


# In[ ]:

# In[234]:


v_differ = [reo[i]-rec[i] for i in range(len(reo))]
print(v_differ)


# In[233]:


plt.figure(figsize=(7,7))
seaborn.displot(v_differ,kde=True).set(title="距离差值分布图")
plt.xlabel("距离差区间")
plt.ylabel("计数")

plt.savefig("distrbution.png",dpi=400)


# In[202]:


data_frame_toget = {
    "维度下标":range(len(rec)),
    '距离差':v_differ
}
data_frame_toget = pd.DataFrame(data_frame_toget)

minimum_values2 = data_frame_toget['距离差'].sort_values(ascending=False).head(32)
print(minimum_values2.values.tolist())


# In[211]:


seaborn.scatterplot(x="维度下标",y="距离差",data=data_frame_toget, color='#aeb6b8').set(title="Gbz与Gbc差值图")
plt.scatter(x=minimum_values2.index.tolist(),y=minimum_values2.values.tolist(), marker='D',color="#20c99b")
plt.savefig("gbzgbc.png",dpi=400)


# In[157]:


seaborn.pairplot(data_frame_toget)


# In[173]:


print(np.argpartition(v_differ, -50)[-50:])


# In[235]:


meaning_dim = np.argpartition(v_differ, -32)[-32:]
meaning_dim


# In[236]:


meaning_dim.sort()
meaning_dim



# In[ ]:


def vec_dim_analysis_spam():
    embedding = bert.get_input_embeddings()
    w1 = 'ham'
    # w2 = 'bad'
    w3 = 'advertisement'
    w1_vec = embedding(torch.tensor(tokenizer.convert_tokens_to_ids(w1)))
    # seaborn.lineplot(w1_vec.detach())
    # w2_vec = embedding(torch.tensor(tokenizer.convert_tokens_to_ids(w2)))
    w3_vec = embedding(torch.tensor(tokenizer.convert_tokens_to_ids(w3)))
    # print(f'{w1}--{w2}:', get_cosine_similarity( w1_vec, w2_vec))  # tensor(0.6438, grad_fn=<DivBackward0>)
    print(f'{w1}--{w3}:', get_cosine_similarity(w1_vec, w3_vec))  # tensor(0.6125, grad_fn=<DivBackward0>)
    # print(f'{w2}--{w3}:', get_cosine_similarity(w2_vec, w3_vec))
    res_o = []
    res_c = []
    for i in range(len(w1_vec)):
        # res_c.append(np.linalg.norm([w1_vec[i].item()-w2_vec[i].item()]))
        res_o.append(np.linalg.norm([w1_vec[i].item()-w3_vec[i].item()]))
    return res_o


# In[ ]:


res_o_spam = vec_dim_analysis_spam()


# In[ ]:


seaborn.scatterplot(res_o_spam)


# In[ ]:


meaning_dim_sapm = np.argpartition(res_o_spam, -16)[-16:]


# In[ ]:


meaning_dim_sapm


# In[ ]:


res_o_spam[789]


# In[ ]:


ord('â')


# In[ ]:


print(tokenizer.get_vocab().get('ham'))


# In[354]:


def word_v(w1, w2, w3):
    embedding = bert.get_input_embeddings()
    w1_vec = embedding(torch.tensor(tokenizer.convert_tokens_to_ids(w1))).detach()
    w2_vec = embedding(torch.tensor(tokenizer.convert_tokens_to_ids(w2))).detach()
    w3_vec = embedding(torch.tensor(tokenizer.convert_tokens_to_ids(w3))).detach()
    return (w1_vec, w2_vec, w3_vec)


w1 = 'negative'
w2 = 'bad'
w3 = 'positive'
w1_vec, w2_vec, w3_vec = word_v(w1, w2, w3)
w1_vec_meaning = [w1_vec[i] for i in meaning_dim]
w2_vec_meaning = [w2_vec[i] for i in meaning_dim]
w3_vec_meaning = [w3_vec[i] for i in meaning_dim]
w1_vec_meaning=torch.tensor(w1_vec_meaning)
w2_vec_meaning=torch.tensor(w2_vec_meaning)
w3_vec_meaning=torch.tensor(w3_vec_meaning)
print(f'{w1}--{w2}:', get_cosine_similarity(w1_vec_meaning, w2_vec_meaning))
print(f'{w1}--{w3}:', get_cosine_similarity(w1_vec_meaning, w3_vec_meaning))
print(f'{w2}--{w3}:', get_cosine_similarity(w2_vec_meaning, w3_vec_meaning))


# In[ ]:


seaborn.pairplot(data_o)

