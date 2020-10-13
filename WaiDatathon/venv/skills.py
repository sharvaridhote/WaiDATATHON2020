import pandas as pd
import numpy as np
import random
from gensim.models.fasttext import FastText
from gensim.test.utils import get_tmpfile
import pickle
from functools import reduce


most_comm = {}
data = pd.read_csv('./user_base.csv' ,converters={'clean_skills': eval})
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]





def construct_skill_map():
    fname = get_tmpfile("fasttext.model")
    model = FastText.load(fname)
    
    skill_map={}
    for each_skill in model.wv.vocab: 
        skill_map[each_skill] = []

    for each_skill in model.wv.vocab:
        for index, each_user in data.iterrows():
            if(each_skill in each_user["clean_skills"]):
                skill_map[each_skill].append(each_user["index"])
    
	

    with open('skill_map.pickle', 'wb') as f:
        pickle.dump(skill_map, f)            
    

def filter_skill(data, user):
    #Merge and pass
    data_pass = data[data['clean_skills'].str.len()==0]
    user_indices_to_match = data[data['clean_skills'].str.len()>0]["index"]
    
    fname = get_tmpfile("fasttext.model")
    model = FastText.load(fname)
    
    
    
    
    for each_skill in user["clean_skills"].values[0]:
        similar_users_list = similar_products(model[each_skill], model)
        # print(similar_users_list)
        most_common(similar_users_list)
    
    indices_found={}
    for user_index in user_indices_to_match:
        if user_index in most_comm:
            indices_found[user_index] = most_comm[user_index]
    
    sorted_indices = [k for k, v in sorted(indices_found.items(), key=lambda item: item[1], reverse=True)]
    print(sorted_indices)
    
    
    return reduce(pd.DataFrame.append, map(lambda i: data[data["index"] == i], sorted_indices))
    # return data[data["index"].isin(sorted_indices)]

def most_common(lists):
    for each_list in lists:
        for each_ele in each_list:
            if each_ele in most_comm:
                most_comm[each_ele] += 1
            else:
                most_comm[each_ele] = 1
    


def similar_products(v, model):
    
    # extract most similar products for the input vector
    top_3_similar = model.similar_by_vector(v)[0:3]
    with open('skill_map.pickle', 'rb') as f:
        skill_map = pickle.load(f)
    # print(skill_map)
    
    # extract name and similarity score of the similar products
    similar_users_list = []
    for sim_word,sim_score in top_3_similar:
        user_list = skill_map[sim_word]
        similar_users_list.append(user_list)
        
    return similar_users_list     
    


def main():
    
    
    skills_train=[]
    for each_skill in data["clean_skills"]:
        if each_skill!=['']:
            skills_train.append(each_skill)
    
    embedding_size = 60
    window_size = 40
    min_word = 5
    down_sampling = 1e-2

    model = FastText(skills_train,
                      size=embedding_size,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                      sg=1,
                      iter=100)

    model.init_sims(replace=True)
    print(model)
    
    fname = get_tmpfile("fasttext.model")
    model.save(fname)
    
 




if __name__ == '__main__':
    main()
    construct_skill_map()