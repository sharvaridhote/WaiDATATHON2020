import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from flask import Flask,request, url_for, redirect, render_template
import numpy as np




app = Flask(__name__)
@app.route('/')
def main():
    return render_template("main.html")

import pandas as pd
from location import filter_location
from skills import filter_skill
from skills import construct_skill_map
from sc_role import filter_sc_role

data = pd.read_csv('./user_base.csv' ,converters={'clean_skills': eval})
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data["Location"] = data["City"]+", "+data["Country"]

#user_id = input()




@app.route('/match' ,methods=['POST'])
# function that takes in Name as input and returns the top 10 matches
def matches():

    user_id = request.form['Name']
    print(user_id)
    user = data.loc[data["index"] == int(user_id)]

    location=request.form['City']

    if not location:
        location=user["Location"].values[0]

    filtered_loc = filter_location(data, user, location)
    print(filtered_loc)

    sorted_skill = filter_skill(filtered_loc, user)
    print(sorted_skill)

    recommended = filter_sc_role(sorted_skill, user)
    print(recommended)

    #Name =  request.form['Name']
    #print(Name)
    recommended_matches = []

    # gettin the index of the Name that matches the name
#    idx = indices[indices == Name].index[0]

    # creating a Series with the similarity scores in descending order
    #score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar Names
    #top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the titles of the best 10 user matches
   # for i in top_10_indexes:
     #   recommended_matches.append(list(df.index)[i])


    return render_template('main.html',pred=recommended)




if __name__ == "__main__":
    app.run()