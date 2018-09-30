import numpy as np
import pandas as pd
import csv

def process(rating_file):

    df = pd.read_csv(rating_file)
    df = df[['userId', 'itemId']]
    df = df.values

    user_list, item_list = [], []

    user_dict = dict()
    item_dict = dict()

    f_user_dict = dict()
    f_item_dict = dict()

    print(len(df))
    for line in df:
        u_id = int(line[0])
        i_id = int(line[1])

        if u_id in user_dict.keys():
            user_dict[u_id].append(i_id)
        else:
            user_dict[u_id] = [i_id]

        if i_id in item_dict.keys():
            item_dict[i_id].append(u_id)
        else:
            item_dict[i_id] = [u_id]

    while True:
        print(len(user_dict.keys()), len(item_dict.keys()))
        flag1, flag2 = True, True

        for u_id in user_dict.keys():
            pos_items = user_dict[u_id]
            valid_items = [idx for idx in pos_items if idx in item_dict.keys()]

            if len(valid_items) >= 10:
                f_user_dict[u_id] = valid_items
            else:
                flag1 = False

        for i_id in item_dict.keys():
            pos_users = item_dict[i_id]
            valid_users = [udx for udx in pos_users if udx in user_dict.keys()]

            if len(valid_users) >= 10:
                f_item_dict[i_id] = valid_users
            else:
                flag2 = False

        user_dict = f_user_dict.copy()
        item_dict = f_item_dict.copy()
        f_user_dict = {}
        f_item_dict = {}

        if flag1 and flag2:
            print('select done')
            break

    remap_user_dict = {}
    u_map_dict = {}
    i_map_dict = {}

    for org_u_id in user_dict.keys():
        if org_u_id not in user_list:
            user_list.append(org_u_id)
        u_id = user_list.index(org_u_id)
        u_map_dict[u_id] = org_u_id

        org_i_ids = user_dict[org_u_id]
        i_ids = []
        for org_i_id in org_i_ids:

            if org_i_id not in item_list:
                item_list.append(org_i_id)
            i_id = item_list.index(org_i_id)
            i_map_dict[i_id] = org_i_id
            
            i_ids.append(i_id)
        remap_user_dict[u_id] = i_ids
    return remap_user_dict, u_map_dict, i_map_dict