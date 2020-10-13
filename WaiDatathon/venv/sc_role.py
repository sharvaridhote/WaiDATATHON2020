import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def filter_sc_role(data, user):
    
    # data.loc[len(data)] = user
    # print(data)
    add_user = False
    val = data[data['index'] == user["index"].values[0]]
    # print(val)
    if val.empty:
        add_user = True

    if add_user:
        data = data.append(user, ignore_index=True)

    print(data)
    data['Experience'] = data['Experience'].astype('int32')

    mean_Experience=int(data.Experience.mean())
    data.loc[data.Experience==0, 'Experience'] = mean_Experience
    data.loc[data.Experience>50, 'Experience'] = mean_Experience
    
    
    
    data['Position_Level'] = data['Position_Level'].map({1:'Senior', 2:'Mid', 3:'Junior'})
    # user['Position_Level'] = user['Position_Level'].map({1:'Senior', 2:'Mid', 3:'Junior'})

    data = data.loc[:,['Name', 'Business_School', 'Position_Level']]

    # data[len(data)] = user[:,['Name', 'Business_School', 'Position_Level']]
    print(data)
    
    data.set_index('Name', inplace = True)
    
    data['bag_of_words'] = ''
    columns = data.columns
    for index, row in data.iterrows():
        words = ''
        for col in columns:
                words = words + row[col]+ ' '
        row['bag_of_words'] = words
        
    data.drop(columns = [col for col in data.columns if col!= 'bag_of_words'], inplace = True)
    
    count = CountVectorizer()
    count_matrix = count.fit_transform(data['bag_of_words'])

    # creating a Series for the Names so they are associated to an ordered numerical
    # list I will use later to match the indexes
    
    indices = pd.Series(data.index)
    # print(indices[:5])
    
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # print(cosine_sim)
    # print("----------------------------------------------")
    
    return matches(user["Name"].values[0], cosine_sim, indices, data)
    

def matches(Name, cosine_sim, indices, data):
    
    recommended_matches = []
    
    
    # gettin the index of the Name that matches the name
    idx = indices[indices == Name].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar Names
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 user matches
    for i in top_10_indexes:
        recommended_matches.append(list(data.index)[i])
        
    return recommended_matches