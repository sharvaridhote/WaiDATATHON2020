def filter_location(data, user, location):
    # print(user["Location"])
    return data.loc[data["Location"].str.match(location)]
