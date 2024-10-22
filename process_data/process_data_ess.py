"""
In this file we read the raw data from EVS2017.sav file and we process
the data according to the section 6 of the article.
"""


import pandas as pd
import csv


def a_adp(element):
    category_dic = {
        4: 1,
        3: 2,
        2: 3,
        1: 4
    }
    try:
        # against the scheme 
        div_value = element['basinc']
        div_support = -(div_value - 2.5) / 1.5

        # for the scheme
        adp_value = category_dic[element['basinc']]
        adp_support = -(adp_value - 2.5) / 1.5
        return adp_support, div_support
    except BaseException:
        return None, None

# Not used in ess
"""
def a_div(element): 
    #We process the data of Q44G ('v155'). never = 1 - 5 strong disagreement
    #INPUT: element = pandas dataframe row
    #Return: value (float [-1,1]) represents a(adp)
    try:
        if element['v155'] == 'never':
            int_value = 1
        elif element['v155'] == 'always':
            int_value = 10
        else:
            int_value = int(element['v155'])
        value = (int_value - 5.5) / 4.5
        return value
    except BaseException:
        return None
"""

def process_principle(dataframe, caseno):
    """
    In this function we count the number of religious or non-religious
    participants per country according to the proceeding Serramià et al. describe
    INPUT: dataframe (pandas dataframe with the results of the EVS 2017)
           country (code of the european country, e.g. ES for Spain)
           caseno (number of case per country)
    Return: tuple (tuple with three values: religious: True if religious
                                            adp: float from -1, 1
                                            div: float from -1, 1)
    """
    # religious or not v6 in EVS2017
    dataframe_row = dataframe[dataframe['idno'] == caseno].iloc[0]
    # smdfslv here is the variable that represents the egalitarianism, but this has been dropped
    egalitarianism = dataframe_row['smdfslv']

    try:
        if egalitarianism == 5 or egalitarianism == 4:
            egalitarian = False
        elif egalitarianism == 2 or egalitarianism == 1:
            egalitarian = True
        else:
            egalitarian = None
    except BaseException:
        egalitarian = None

    if egalitarian is None:
        return None
    else:
        return (egalitarian)


def process_participant(dataframe, caseno):
    """
    In this function we count the number of religious or non-religious
    participants per country according to the proceeding Serramià et al. describe
    INPUT: dataframe (pandas dataframe with the results of the EVS 2017)
           country (code of the european country, e.g. ES for Spain)
           caseno (number of case per country)
    Return: tuple (tuple with three values: religious: True if religious
                                            adp: float from -1, 1
                                            div: float from -1, 1)
    """
    # religious or not v6 in EVS2017
    dataframe_row = dataframe[dataframe['idno'] == caseno].iloc[0]
    universalism = dataframe_row['imptrad']
    hedonism = dataframe_row['ipgdtim']

    """
    try:
        if universalism == 6 or universalism == 5 or universalism == 4:
            universalist = False
        elif universalism == 3 or universalism == 2 or universalism == 1:
            universalist = True
        else:
            universalist = None
    except BaseException:
        universalist = None
    """

    try:
        if universalism > hedonism:
            universalist = True
        else:
            universalist = False
    except BaseException:
        universalist = None

    # we compute a_ad and a_dv
    action_adp, action_div = a_adp(dataframe_row)

    if universalist is None or action_adp is None or action_div is None:
        return None
    else:
        return (universalist, action_adp, action_div)


def process_country(dataframe, country):
    """
    Process information for each country
    INPUT: dataframe (pandas dataframe with the results of the EVS 2017)
           country (code of the european country, e.g. ES for Spain)
    Return: dict with # of religious and non-religious citizens,
            a_rl(ad), a_pr(ad), a_rl(dv) and a_pr(dv)
    """
    df = dataframe[dataframe['cntry'] == country]
    n_row = df.shape[0]
    # setting counters to compute the mean of each judgement value:
    # a_rl(ad), a_pr(ad), a_rl(dv) and a_pr(dv)
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

    for i in range(0, n_row):
        caseno = df.iloc[i]['idno']
        tuple_ = process_participant(df, caseno)  # information of the case
        if tuple_:
            if tuple_[0]:  # True for religious citizens
                n_religious += 1
                # ignore missing data
                if tuple_[1] is not None:  # adopt judgement
                    n_rel_adp += 1
                    sum_a_adp_rel += tuple_[1]
                if tuple_[2] is not None:  # divorce judgement
                    n_rel_div += 1
                    sum_a_div_rel += tuple_[2]
            else:  # non-religious citizens
                n_nonreligious += 1
                if tuple_[1] is not None:  # adopt judgement
                    n_nonrel_adp += 1
                    sum_a_adp_nonrel += tuple_[1]
                if tuple_[2] is not None:  # divorce judgement
                    n_nonrel_div += 1
                    sum_a_div_nonrel += tuple_[2]
        else:
            continue
    print("country", country)
    print('n_religious: ', n_religious)
    print('n_nonreligious: ', n_nonreligious)
    print('n_rel_adp: ', n_rel_adp)
    print('n_nonrel_adp: ', n_nonrel_adp)
    print('n_rel_div: ', n_rel_div)
    print('n_nonrel_div: ', n_nonrel_div)
    print('sum_a_adp_rel: ', sum_a_adp_rel)
    print('sum_a_adp_nonrel: ', sum_a_adp_nonrel)
    print('sum_a_div_rel: ', sum_a_div_rel)
    print('sum_a_div_nonrel: ', sum_a_div_nonrel)
    return {
        'rel': n_religious,
        'nonrel': n_nonreligious,
        'a_adp_rel': sum_a_adp_rel / n_rel_adp,
        'a_adp_nonrel': sum_a_adp_nonrel / n_nonrel_adp,
        'a_div_rel': sum_a_div_rel / n_rel_div,
        'a_div_nonrel': sum_a_div_nonrel / n_nonrel_div
    }

def principle_process_country(dataframe, country):
    """
    Process information for each country
    INPUT: dataframe (pandas dataframe with the results of the EVS 2017)
           country (code of the european country, e.g. ES for Spain)
    Return: dict with # of religious and non-religious citizens,
            a_rl(ad), a_pr(ad), a_rl(dv) and a_pr(dv)
    """
    df = dataframe[dataframe['cntry'] == country]
    n_row = df.shape[0]
    # setting counters to compute the mean of each judgement value:
    # a_rl(ad), a_pr(ad), a_rl(dv) and a_pr(dv)
    n_religious = 0
    n_nonreligious = 0

    for i in range(0, n_row):
        caseno = df.iloc[i]['idno']
        value = process_principle(df, caseno)  # information of the case
        if value:
            n_religious += 1
        else:  # non-religious citizens
            n_nonreligious += 1

    print("country", country)
    print('n_religious: ', n_religious)
    print('n_nonreligious: ', n_nonreligious)
    return {
        'rel': n_religious,
        'nonrel': n_nonreligious,
    }

if __name__ == '__main__':
    df = pd.read_spss("ESS8e02_3-subset.sav", convert_categoricals=False)
    # Subset here is 
    df = df.dropna(subset=['imptrad', 'ipgdtim', 'basinc'])


    # we create a dictionary to store the data per country
    dictionary = {}
    for country in list(df['cntry'].unique()):
        dict_ = process_country(
            df[['cntry', 'idno', 'imptrad', 'ipgdtim', 'basinc']], country)
        dictionary.update({country: dict_})
    columns = ['country']
    for key in dictionary[country].keys():
        columns.append(key)
    csv_rows = [columns]
    for country in dictionary.keys():
        csv_rows2 = [country]
        for item in dictionary[country].keys():
            csv_rows2.append(dictionary[country][item])
        csv_rows.append(csv_rows2)
    # we store the data in a file
    with open('processed_data_ess.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
