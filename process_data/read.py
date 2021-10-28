# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 17:10:55 2021

@author: ROGER
"""

"""
In this file we will read the raw data from EVS2017.sav file and we will process 
the data according to the article "Encoding ethics and moral values".
"""

import pandas as pd
import csv

#df = pd.read_spss("EVS2017.sav")

#%%

def a_adp(element):
    """
    We process the data of Q27A ('v82'). 1 strong agreement - 5 strong disagreement
    INPUT: element = pandas dataframe row 
    Return: value (float [-1,1]) represents a(adp)
    """
    category_dic = {'agree strongly':1,'agree':2,'neither agree nor disagree':3,
                    'disagree':4,'disagree strongly':5}
    try:
        int_value = category_dic[element['v82']]
        value = -(int_value-3)/2
        return value
    except:
        return None
    
def a_div(element):
    """
    We process the data of Q44G ('v155'). never = 1 - 5 strong disagreement
    INPUT: element = pandas dataframe row 
    Return: value (float [-1,1]) represents a(adp)
    """
    try:
        if element['v155'] == 'never':
            int_value = 1
        elif element['v155'] == 'always':
            int_value = 10
        else:
            int_value = int(element['v155'])
            
        value = (int_value-5.5)/4.5
        return value
    except:
        return None


def process_participant(dataframe,caseno):
    """
    In this function we count the number of religious or non-religious 
    participants per country according to the proceeding Serramià et al. describe
    INPUT: dataframe (pandas dataframe with the results of the EVS 2017)
           country (code of the european country, e.g. ES for Spain)
           caseno (number of case per country)
    Return: tuple (tuple with three values: religious: True if religious
                        adp: float from -1,1
                        div: float from -1,1)
    """
    #religious or not v6 in EVS2017
    dataframe_row = dataframe[dataframe['caseno']==caseno].iloc[0]
    religiousness = dataframe_row['v6']
    try:
        if religiousness == 'not at all important' or religiousness == 'not important':
            religious = False
        elif religiousness == 'quite important' or religiousness == 'very important':
            religious = True
        else:
            religious = None
    except:
        religious = None

    action_adp = a_adp(dataframe_row)
    action_div = a_div(dataframe_row)
    
    if religious == None or action_adp == None or action_div == None:
        return None
    else:
        return (religious,action_adp,action_div)
    
def process_country(dataframe,country):
    """
    Process information for each country
    INPUT: dataframe (pandas dataframe with the results of the EVS 2017)
           country (code of the european country, e.g. ES for Spain)
    Return: dict
    """
    df = dataframe[dataframe['c_abrv']==country]
    n_row = df.shape[0]
    n_religious = 0
    n_nonreligious = 0
    sum_a_adp_rel = 0
    sum_a_adp_nonrel = 0
    sum_a_div_rel = 0
    sum_a_div_nonrel = 0
    for i in range(0,n_row):
        caseno = df.iloc[i]['caseno']
        tuple_ = process_participant(df,caseno)
        if tuple_:
            if tuple_[0]:
                n_religious += 1
                sum_a_adp_rel += tuple_[1]
                sum_a_div_rel += tuple_[2]
            else:
                n_nonreligious += 1
                sum_a_adp_nonrel += tuple_[1]
                sum_a_div_nonrel += tuple_[2]
        else:
            continue
        
    return {'rel':n_religious,'nonrel':n_nonreligious,
            'a_adp_rel':sum_a_adp_rel/n_religious,'a_adp_nonrel':sum_a_adp_nonrel/n_nonreligious,
            'a_div_rel':sum_a_div_rel/n_religious,'a_div_nonrel':sum_a_div_nonrel/n_nonreligious}
    

def process_participant_v2(dataframe,caseno):
    """
    In this function we count the number of religious or non-religious 
    participants per country according to the proceeding Serramià et al. describe
    INPUT: dataframe (pandas dataframe with the results of the EVS 2017)
           country (code of the european country, e.g. ES for Spain)
           caseno (number of case per country)
    Return: tuple (tuple with three values: religious: True if religious
                        adp: float from -1,1
                        div: float from -1,1)
    """
    #religious or not v6 in EVS2017
    dataframe_row = dataframe[dataframe['caseno']==caseno].iloc[0]
    religiousness = dataframe_row['v6']
    try:
        if religiousness == 'not at all important' or religiousness == 'not important':
            religious = False
        elif religiousness == 'quite important' or religiousness == 'very important':
            religious = True
        else:
            religious = None
    except:
        religious = None

    action_adp = a_adp(dataframe_row)
    action_div = a_div(dataframe_row)
    
    if religious == None:
        return None
    else:
        return (religious,action_adp,action_div)
    
def process_country_v2(dataframe,country):
    """
    Process information for each country
    INPUT: dataframe (pandas dataframe with the results of the EVS 2017)
           country (code of the european country, e.g. ES for Spain)
    Return: dict
    """
    df = dataframe[dataframe['c_abrv']==country]
    n_row = df.shape[0]
    n_religious = 0
    n_nonreligious = 0
    n_rel_adp = 0
    n_nonrel_adp = 0
    n_rel_div = 0
    n_nonrel_div = 0
    sum_a_adp_rel = 0
    sum_a_adp_nonrel = 0
    sum_a_div_rel = 0
    sum_a_div_nonrel = 0
    for i in range(0,n_row):
        caseno = df.iloc[i]['caseno']
        tuple_ = process_participant_v2(df,caseno)
        if tuple_:
            if tuple_[0]:
                n_religious += 1
                if tuple_[1] != None:
                    n_rel_adp += 1
                    sum_a_adp_rel += tuple_[1]
                if tuple_[2] != None:
                    n_rel_div += 1
                    sum_a_div_rel += tuple_[2]
            else:
                n_nonreligious += 1
                if tuple_[1] != None:
                    n_nonrel_adp += 1
                    sum_a_adp_nonrel += tuple_[1]
                if tuple_[2] != None:
                    n_nonrel_div += 1
                    sum_a_div_nonrel += tuple_[2]
        else:
            continue
        
    return {'rel':n_religious,'nonrel':n_nonreligious,
            'a_adp_rel':sum_a_adp_rel/n_rel_adp,'a_adp_nonrel':sum_a_adp_nonrel/n_nonrel_adp,
            'a_div_rel':sum_a_div_rel/n_rel_div,'a_div_nonrel':sum_a_div_nonrel/n_nonrel_div}

    
if __name__ == '__main__':
    df = pd.read_spss("EVS2017.sav")
    #process_country(df,'AL')
    dictionary = {}
    for country in list(df['c_abrv'].unique()):
        dict_ = process_country_v2(df[['c_abrv','caseno','v6','v82','v155']],country)
        dictionary.update({country:dict_})
        
    columns = ['country']
    for key in dictionary[country].keys():
        columns.append(key)
    
    csv_rows = [columns]
    for country in dictionary.keys():
        csv_rows2 = [country]
        for item in dictionary[country].keys():
            csv_rows2.append(dictionary[country][item])
        csv_rows.append(csv_rows2)
    
    with open('AAMAS2022.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
    
    
    