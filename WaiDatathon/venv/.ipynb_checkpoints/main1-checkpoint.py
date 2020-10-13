import pandas as pd
from location import filter_location
from skills import filter_skill
from skills import construct_skill_map
from sc_role import filter_sc_role

data = pd.read_csv('./user_base.csv' ,converters={'clean_skills': eval})
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data["Location"] = data["City"]+", "+data["Country"]

user_id = input()
user = data.loc[data["index"]==int(user_id)]

filtered_loc = filter_location(data,user)
print(filtered_loc)


sorted_skill = filter_skill(filtered_loc,user)
print(sorted_skill)

recommended = filter_sc_role(sorted_skill,user)
print(recommended)


