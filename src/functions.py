import os
import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import certifi
import urllib3
import re
from csv import DictReader, DictWriter
import datetime as dt
import glob
import datetime
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 





'''This file holds the functions that I will be using
for the project. You can find the docstrings for each
function below the line of code defining the function.
There are functions that look extremely similar but were
necessary due to the different formats in the .csv files
used.'''



################################################################


def evaluation(y, y_hat, title = 'Confusion Matrix'):
    
    '''takes in true values and predicted values.
    The function then prints out a classifcation report
    as well as a confusion matrix using seaborn's heatmap.'''
    
    cm = confusion_matrix(y, y_hat)
    precision = precision_score(y, y_hat)
    recall = recall_score(y, y_hat)
    accuracy = accuracy_score(y,y_hat)
    print(classification_report(y, y_hat))
    print('Recall: ', recall)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    sns.heatmap(cm,  cmap= 'Greens', annot=True, fmt='d')
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.title(title)
    
    plt.show()



################################################################
def plot_corr(df):
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(10,8))
    ax = sns.heatmap(df.corr(),linewidths=.2, mask=mask)
    plt.show();



def content_cleaner(df):
    
    '''This function takes in a dataframe and
    converts the rank feature to numerical and
    formats the name feature.'''
    
    df = df.dropna(axis=0)
    df['rank'] = [x.strip('\n') for x in df['rank']]
    df['rank'] = [x.strip('\n') for x in df['rank']]
    df['rank'] = [x.replace('G', '1.') for x in df['rank']]
    df['rank'] = [x.replace('S', '2.') for x in df['rank']]
    df['rank'] = [x.replace('B', '3.') for x in df['rank']]
    
    
    df['name'] = [x.strip('\n\n\n\n\n\n\n\n\n\n\n\n') for x in df['name']]
    df['name'] = [x.replace('\n', ' ') for x in df['name']]
    df['name'] = [x[:-3] for x in df['name']]
    df['name'] = [x.title() for x in df['name']]
    
    df['result'] = [x.strip() for x in df['result']]

    return df


################################################################

def olympic_query(loc, year, event):
    
    '''This function is to send a query and
    fetch the html text for olympic.org'''
    
    url = 'https://www.olympic.org/{}-{}/athletics/{}'.format(loc, year, event)
    req = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                             ca_certs=certifi.where())
    res = req.request('GET', url)
    soup = BeautifulSoup(res.data, 'html.parser')
    contents = soup.find_all(class_ = 'table4')
    return contents


################################################################

def olympic_scraper(content_):
    
    '''This function takes in the html text
    from olympic.org and returns a dataframe
    with the data'''

    a = []
    b = []
    c = []
    d = []

    tables = []
    for x in content_:
        rows = x.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) == 4:
                a.append(cells[0].text)
                b.append(cells[1].text)
                c.append(cells[2].text)
                d.append(cells[3].text)

    df = pd.DataFrame(a, columns=['rank'])
    df['name'] = b
    df['result'] = c
    
    df.name = [x.strip() for x in df.name]
    

    return df


################################################################

def preprocess_df(df, cont_feat_list, cat_feat_list):
    
    '''This function creates the dataframe
    used for the baseline model. The function'''
    
    X = df.drop(['flagged'], axis=1)
    y = df.flagged
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=.2)
    
    X_train_cont = X_train[cont_feat_list].astype(float)
    X_train_cat = X_train[cat_feat_list]
    X_test_cont = X_test[cont_feat_list].astype(float)
    X_test_cat = X_test[cat_feat_list]

    X_train_index = X_train.index
    X_test_index = X_test.index

    ss = StandardScaler()
    X_train_scaled = pd.DataFrame(ss.fit_transform(X_train_cont), 
                                  columns=X_train[cont_feat_list].columns, 
                                  index=X_train_index)
    X_test_scaled = pd.DataFrame(ss.transform(X_test_cont), 
                                 columns=X_test[cont_feat_list].columns, 
                                 index=X_test_index)
    ohe = OneHotEncoder()
    X_train_encoded = ohe.fit_transform(X_train_cat)
    X_test_encoded = ohe.transform(X_test_cat)

    train_columns = ohe.get_feature_names(input_features=X_train_cat.columns)
    test_columns = ohe.get_feature_names(input_features=X_test_cat.columns)

    X_train_processed = pd.DataFrame(X_train_encoded.todense(), columns=train_columns, index=X_train_index)
    X_test_processed = pd.DataFrame(X_test_encoded.todense(), columns=test_columns, index=X_test_index)

    X_train_all = pd.concat([X_train_scaled, X_train_processed], axis=1)
    X_test_all = pd.concat([X_test_scaled, X_test_processed], axis=1)
    
    return X_train_all, X_test_all, y_train, y_test
    
################################################################

def create_athlete_df():
    
    '''This function fetches all .csv files
    containing the athletes per event during 
    the 2004-2016 Summer Olympic Games. The 
    function reads in each csv file and appends
    the dataframe to the dfs list. Then loops 
    through the dfs list and appends each
    dataframe to return one full dataframe.'''
    
    
    dfs = []
    csvs = glob.glob('../data/athletes_by_event/*')
    for csv in csvs:
        #print('loading_csv:{}'.format(csv))
        df_ = pd.read_csv(csv, delimiter='\t', keep_default_na=False)
        df_ = df_.drop('Unnamed: 5', axis=1)
        dfs.append(df_)
    if len(dfs) == 40:
        print('finished')
    df = dfs[0]
    for x in range(len(dfs)-1):

        df = df.append(dfs[x])

    df.columns = [x.lower() for x in df.columns]
    df.name = [x.title() for x in df.name]
    df = df.drop_duplicates()

    df = df.sort_values(by='name')

    return df

################################################################

def wiki_scraper(content):
    
    '''This function takes in html text and tags
    returning a dataframe'''
    
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    g = []
    tables = []
    for table in content:
        tables.append(table)
        for table_ in tables:
            rows = table_.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) == 7:
                    a.append(cells[0].text)
                    b.append(cells[1].text)
                    c.append(cells[2].text)
                    d.append(cells[3].text)
                    e.append(cells[4].text)
                    f.append(cells[5].text)
                    g.append(cells[6].text)
    df = pd.DataFrame(a, columns=['name'])
    df['country'] = b
    df['event'] = c
    df['date_of_violation'] = d
    df['substance'] = e
    df['sanction'] = f
    df['references'] = g
    
    return df

################################################################


def create_doping_df():
    
    '''This function fetches the .pdf files from
    the 2020 Sanctions List and appends them
    together returning a complete dataframe
    containing all athletes involved in doping
    irregularities. Next, preprocesses the dataframe
   
    (formatting names, dropping rows with
    holding irrelevant values)'''
    
    doping_dfs = tabula.read_pdf('../data/datasets/July-2020-Sanctions-List-Full.pdf', pages='all',
                     output_format='dataframe')
    df = doping_dfs[0]
    for x in range(len(doping_dfs)-1):
        df = df.append(doping_dfs[x])
    df.columns = ['name', 'dob', 'nationality', 'roll',
                 'sex', 'discipline_1', 'discipline_2', 
                 'date_of_infraction', 'sanction', 
                 'ineligibility_until', 'dq_date', 
                 'lifetime_ban', 'infraction_type', 
                 'adrv_rules', 'adrv_notes', 'description']
    
    df = df[df['infraction_type'] == 'Doping']
    df = df.rename(columns={'infraction_type':'flagged'})
    df.name = [x.title() for x in df.name]
    df.name = [x.strip(',') for x in df.name]
    
    df.dq_date = df.dq_date.fillna(value='0')
    df.dq_date = [x.strip('Since') for x in df.dq_date]
    df.dq_date = [x.strip('From') for x in df.dq_date]
    
    df = df.drop([33,41,72,85,94,106,111,121,122,123,124,126,148,149,
                     160,186,204,214,297,305,315,332,351,352,363,373,375,
                     379,381,385,392,400,402,409,411,418,436,437,446,472,
                     483,484,485,489,490,500,505], axis=0)
    df.reset_index(drop=True, inplace=True)
    
    df.dq_date = [str(x) for x in df.dq_date]
    df.dq_date = [x.replace('.', '/') for x in df.dq_date]
    df.dq_date = [x.replace('-', '/') for x in df.dq_date]
    
    dates = []
    for x in df.dq_date:
        dates.append('20'+x[-2:])
    df.dq_date = [x for x in dates]
    names = []
    for x in df.name:
        split = x.split()
        if len(split) == 2:
            x = (split[-1] + ' ' + split[0])
        elif len(split) == 3:
            x = (split[1] + ' ' + split[-1] + ' ' + split[0])
        elif len(split) == 4:
            if (',') in split[1]:
                x = (split[2] + ' ' + split[-1] + ' ' + split[0] + ' ' + split[1])
            else:
                x = (split[-1] + ' ' + split[0] + ' ' + split[1] + ' ' + split[2])
        else:
            x = (split[0])
        names.append(x)

    df.name = [x for x in names]
    df.name = [x.replace(',','') for x in df.name]
    df.name = [x.replace('(','').replace(')','') for x in df.name]

    
    df = df.sort_values(by='name')
    print('finished')
    
    return df

################################################################


def col_format(df):
    
    '''This function takes in a dataframe, formats
    the features and removes the whitespace around
    the values.'''
    
    df['name'] = [x.strip('\n') for x in df['name']]
    df['country'] = [x.strip('\n') for x in df['country'].values]
    df['date_of_violation'] = [x.strip('\n') for x in df['date_of_violation'].values]
    df['event'] = [x.strip('\n') for x in df['event'].values]
    df['substance'] = [x.strip('\n') for x in df['substance'].values]
    df['sanction'] = [x.strip('\n') for x in df['sanction'].values]
    df['references'] = [x.strip('\n') for x in df['references'].values]
    return df

################################################################


def create_wiki_doping():
    
    '''This function creates the doping 
    dataframe containing athletes involved
    in doping irregularities (obtained from
    wikipedia)
    The functions fetches the html text and tags
    from the url and creates a dataframe using the 
    wiki_scraper function above. Then formats the 
    column values using the col_format function
    above. Finally, removing rows holding irrelevant
    values and replacing names with different spellings
    than in athlete dataframe with the appropriate spellings
    to match.'''
    
    url = 'https://en.wikipedia.org/wiki/List_of_doping_cases_in_athletics'
    req = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                         ca_certs=certifi.where())
    res = req.request('GET', url)
    soup = BeautifulSoup(res.data, 'html.parser')
    contents = soup.find_all('table', class_='wikitable sortable')  
    
    
    wiki_doping = wiki_scraper(contents)

    wiki_doping = col_format(wiki_doping)

    wiki_doping.date_of_violation = [x.replace('23 May 2002  24 June 2002', '2002') for x in wiki_doping.date_of_violation]
    wiki_doping.date_of_violation = [x[:4] for x in wiki_doping.date_of_violation]

    drop_value = wiki_doping.date_of_violation[44]

    wiki_doping = wiki_doping[wiki_doping.date_of_violation != drop_value]
    wiki_doping = wiki_doping[wiki_doping.date_of_violation != 'Unkn']

    rows_to_drop = [0, 60, 197, 390, 632, 905, 1198, 1532, 1874, 1907, 2249,
                    2293, 2635, 2697, 3039, 3173, 3515, 3693, 4035, 4278, 4289,
                    4631, 4874, 4902, 5244, 5487, 5536, 5878, 6121, 6196, 6212,
                    6554, 6797, 6872, 6889, 7231, 7474, 7549, 7603, 7945, 8188,
                    8263, 8392, 8734, 8977, 9052, 9219, 9561, 9804, 9879, 10052,
                    10394, 10637, 10712, 10907, 11249, 11492, 11567, 11789, 12131,
                    12374, 12449, 12672, 13014, 13257, 13332, 13572, 13914, 14157, 14232]

    wiki_doping = wiki_doping.drop(rows_to_drop, axis=0)

    wiki_doping.date_of_violation = [int(x) for x in wiki_doping.date_of_violation]

    wiki_doping.drop_duplicates(inplace=True)


    wiki_names = list(wiki_doping.name.unique())

    wiki_doping = wiki_doping.sort_values(by='name')

    wiki_doping = wiki_doping[wiki_doping.date_of_violation > 2003]

    wiki_doping = wiki_doping[wiki_doping.date_of_violation < 2017]

    wiki_doping.reset_index(drop=True, inplace=True)
    
    names_to_replace = {'Abderrahime Bouramdane': 'Abderrahhime Bouramdane',
                     'Aleksey Lesnichiy': 'Aleksey Lesnichy',
                     'Aliona Dubitskaya': 'Alyona Dubitskaya',
                     'Amina Aït Hammou': 'Mina Aït Hammou',
                     'Anastasios Gousis': 'Tasos Gousis',
                     'Anastassya Kudinova': 'Anastasiya Kudinova',
                     'Anil Kumar': 'Anil Kumar Sangwan',
                     'Anis Ananenka': 'Anis Ananenko',
                     'Antonio David Jiménez': 'Antonio Jiménez',
                     'Aslı Çakır Alptekin': 'Aslı Çakır',
                     'Ato Stephens': 'Ato Modibo',
                     'Aziz Zakari': 'Abdul Aziz Zakari',
                     'Bernard Williams': 'Bernard Williams Iii',
                     'Boštjan Šimunic': 'Boštjan Šimunič',
                     'Bruno de Barros': 'Bruno Lins',
                     'Béranger-Aymard Bossé': 'Béranger Bosse',
                     'Christopher Williams': 'Chris Williams',
                     'Dimitrios Chondrokoukis': 'Dimitrios Khondrokoukis',
                     'Dmytro Kosynskyy': 'Dmytro Kosynskiy',
                     'Dzmitry Marshin': 'Dmitri Marşin',
                     'Elena Arzhakova': 'Yelena Arzhakova',
                     'Esref Apak': 'Eşref Apak',
                     'Folashade Abugan': 'Shade Abugan',
                     'Fouad Elkaam (Fouad El Kaam)': 'Fouad El-Kaam',
                     'Ghfran Almouhamad': 'Ghfran Mouhamad',
                     'Helder Ornelas': 'Hélder Ornelas',
                     'Hind Dehiba': 'Hind Déhiba',
                     'Iryna Yatchenko': 'Irina Yatchenko',
                     'Ivan Emilianov': 'Ion Emilianov',
                     'Jillian Camarena-Williams': 'Jill Camarena-Williams',
                     'Joyce Zakari Joy Nakhumicha Sakari': 'Joy Nakhumicha Sakari',
                     'Karin Mey Melis': 'Karin Melis',
                     'Katsiaryna Artsiukh': 'Yekaterina Belanovich',
                     'Konstadinos Baniotis': 'Kostas Baniotis',
                     'Konstadinos Filippidis': 'Kostas Filippidis',
                     'LaShawn Merritt': 'Lashawn Merritt',
                     'LaVerne Jones-Ferrette': 'Laverne Jones',
                     'Lü Huihui': 'Lu Huihui',
                     'Marina Marghieva': 'Marina Marghiev-Nicișenco',
                     'Michalis Stamatogiannis': 'Mikhalis Stamatogiannis',
                     'Mohamed El Hachimi': 'Mohamed El-Hachimi',
                     'Mounira Al-Saleh': 'Monira Alsaleh',
                     'Musa Amer Obaid': 'Musa Amer',
                     'Nadzeya Ostapchuk': 'Nadezhda Ostapchuk',
                     'Nataliia': 'Nataliya Lupu',
                     'Neelam Jaswant Singh': 'Neelam Jaswant Singh Dogra',
                     'Nevin Yanit': 'Nevin Yanıt',
                     'Oludamola Osayomi': 'Damola Osayomi',
                     'Olutoyin Augustus  (Toyin Augustus)': 'Toyin Augustus',
                     'Pavel Kryvitski': 'Pavel Krivitsky',
                     'Rachid Ghanmouni': 'Rachid El-Ghanmouni',
                     'Rosa America Rodríguez': 'Rosa Rodríguez',
                     'Roxana Bârcă': 'Roxana Elisabeta Bîrcă',
                     'Ruddy Zang Milama': 'Ruddy Zang-Milama',
                     'Semiha Mutlu Semiha Metlu-Ozdemir': 'Semiha Mutlu',
                     'Shawnacy Barber': 'Shawn Barber',
                     'Shelly-Ann Fraser': 'Shelly-Ann Fraser-Pryce',
                     'Stanislav Emelyanov': 'Stanislav Melnykov',
                     'Sultan Al-Dawoodi': 'Sultan Mubarak Al-Dawoodi',
                     'Tatiana Aryasova': 'Tatyana Aryasova',
                     'Tetiana Petlyuk': 'Tetiana Petliuk',
                     'Tezzhan Naimova': 'Tezdzhan Naimova',
                     'Vania Stambolova': 'Vanya Stambolova',
                     'Venelina Veneva': 'Venelina Veneva-Mateeva',
                     'Zalina Marghieva': 'Zalina Marghiev',
                     'Zohar Zimro': 'Zohar Zemiro'}
    
    wiki_doping.name = wiki_doping.name.replace(names_to_replace) 
    
    return wiki_doping

################################################################

def create_results_df():
    
    '''This function retrieves all the .csv
    files containing the event results for the 
    Summer Olympic Games, reads them in as dataframes
    and appends the dataframes to the dfs list.
    Next, will loop through the list of dataframes
    and append them together returning a complete
    dataframe sorted by the Athlete name
    
    The data format for some of the event results
    were not in a consistent format. For those 
    events this function could not be used.'''
    
    dfs = []
    csvs = glob.glob('../data/results_by_event/*')
    for csv in csvs:
        #print('loading_csv:{}'.format(csv))
        df_ = pd.read_csv(csv, delimiter='\t', keep_default_na=False)
        dfs.append(df_)

    df = dfs[0]
    for x in range(len(dfs)-1):

        df = df.append(dfs[x])


    df = df.sort_values(by='Athlete')

    return df

################################################################


def create_100m_results():
    
    '''Function to create a dataframe holding
    the results from the 100m dash events
    for both men and women during the years
    2004-2016. The function fetches all the
    .csv files in the 100m_dash folder, combines
    them all into one dataframe and performs
    formatting steps on the columns and values.
    '''
    
    dfs = []
    csvs = glob.glob('../data/results_by_event/100m_dash/*')
    for csv in csvs:
        #print('loading_csv:{}'.format(csv))
        df_ = pd.read_csv(csv, delimiter='\t', keep_default_na=False)
        df_.insert(10, 'games', '20{}'.format(csv[-6:-4]))
        dfs.append(df_)

    df = dfs[0]
    for x in range(len(dfs)-1):

        df = df.append(dfs[x])

    df.columns = [x.lower() for x in df.columns]
    df.columns = [x.replace(' ','_') for x in df.columns]
    df.athlete = [x.title() for x in df.athlete]
    df = df.drop(['pos' , 'nr', 'preliminary_round',
                  'unnamed:_8', 'unnamed:_9', 'unnamed:_10'], axis=1)
    df.r1 = [x[:5] for x in df.r1]
    df.qf = [x[:5] for x in df.r1]
    df.sf = [x[:5] for x in df.sf]
    df.final = [x[:5] for x in df.final]
    
    df = df[['athlete', 'noc', 'games', 'r1', 'qf', 'sf', 'final']]   
    
    df.reset_index(drop=True, inplace=True)
    value = df.final[9]
    df = df.replace(value,'0')
    df = df.fillna(value='0')
    df = df.replace("'", " ")
    df = df.replace('– (AC','0')
    x = df.sf[161]
    df.sf = df.sf.replace(x,'0')
    df.final = df.final.replace(x, '0')
    
    df.r1 = [float(x) for x in df.r1]
    df.qf = [float(x) for x in df.qf]
    df.sf = [float(x) for x in df.sf]
    df.final = [float(x) for x in df.final]
    df = df.sort_values(by='games')
    df.reset_index(drop=True, inplace=True)
    
    return df

################################################################

def create_200m_results():
    
    '''Function to create a dataframe holding
    the results from the 200m dash events
    for both men and women during the years
    2004-2016. The function fetches all the
    .csv files in the 100m_dash folder, combines
    them all into one dataframe and performs
    formatting steps on the columns and values.
    '''
    
    dfs = []
    csvs = glob.glob('../data/results_by_event/200m_dash/*')
    for csv in csvs:
        #print('loading_csv:{}'.format(csv))
        df_ = pd.read_csv(csv, delimiter='\t', keep_default_na=False)
        df_.insert(10, 'games', '20{}'.format(csv[-6:-4]))

        dfs.append(df_)

    df = dfs[0]
    for x in range(len(dfs)-1):

        df = df.append(dfs[x])

    df.columns = [x.lower() for x in df.columns]
    df.columns = [x.replace(' ','_') for x in df.columns]
    df.athlete = [x.title() for x in df.athlete]
    df = df.drop(['pos' , 'nr', 'unnamed:_7',
               'unnamed:_8', 'unnamed:_9', 'unnamed:_10'], axis=1)
    df.r1 = [x[:5] for x in df.r1]
    df.qf = [x[:5] for x in df.r1]
    df.sf = [x[:5] for x in df.sf]
    df.final = [x[:5] for x in df.final]

    df = df[['athlete', 'noc', 'games', 'r1', 'qf', 'sf', 'final']]

    df.reset_index(drop=True, inplace=True)
    value = df.final[9]
    df = df.replace(value,'0')
    df = df.fillna(value='0')
    df = df.replace("'", " ")
    df = df.replace('– (AC','0')
    df = df.replace('– (DN','0')


    df.r1 = [float(x) for x in df.r1]
    df.qf = [float(x) for x in df.qf]
    df.sf = [float(x) for x in df.sf]
    df.final = [float(x) for x in df.final]
    df = df.sort_values(by='athlete')
    df = df.rename(columns={'athlete':'name'})
    
    df.reset_index(drop=True, inplace=True)


    return df

################################################################


def world_athletics_scraper(content):
    
    '''This function takes in a list html text
    and tags contents and creates a dataframe
    by looping through the tables in
    the contents list and appending the rows
    to a new dataframe.'''
    a = []
    b = []
    c = []
    d = []
    e = []

    tables = []
    for table in content:
        tables.append(table)
        for table_ in tables:
            rows = table_.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) == 5:
                    a.append(cells[0].text)
                    b.append(cells[1].text)
                    c.append(cells[2].text)
                    d.append(cells[3].text)
                    e.append(cells[4].text)

    df = pd.DataFrame(a, columns=['pos'])
    df['name'] = b
    df['country'] = c
    df['time'] = d
    df['reaction_time'] = e
    
    df.pos = [x.strip() for x in df.pos]
    df.name = [x.strip() for x in df.name]
    df.country = [x.strip() for x in df.country]
    df.time = [x.strip() for x in df.time]
    df.reaction_time = [x.strip() for x in df.reaction_time]
    df = df.drop_duplicates()

    return df

################################################################


def world_athletics_query():
    
    '''This function fetches the html text
    and tags from worldathletics.org,'''
    
    url = 'https://www.worldathletics.org/results/olympic-games/2004/28th-olympic-games-6913163/men/100-metres/heats/result'
    req = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                             ca_certs=certifi.where())
    res = req.request('GET', url)
    soup = BeautifulSoup(res.data, 'html.parser')
    contents = soup.find_all(class_ = 'table-wrapper')
    return contents

################################################################


def create_hurdle_results():
    
    '''Function to create a dataframe holding
    the results from the Hurdle events
    for both men and women during the years
    2004-2016. The function fetches all the
    .csv files in the 100mWomens_110mMens_hurdles
    folder, combines them all into one dataframe 
    and performs formatting steps on the columns 
    and values.
    Some values in the r1, qf, sf, and final
    columns contain strings indicating if an 
    athlete was disqualified for doping or
    something that happened during the event.
    The string are removed only leaving the event results
    '''
    
    
    dfs = []
    csvs = glob.glob('../data/results_by_event/100mWomens_110mMens_hurdles/*')
    for csv in csvs:
        #print('loading_csv:{}'.format(csv))
        df_ = pd.read_csv(csv, delimiter='\t', keep_default_na=False)
        df_.insert(10, 'games', '20{}'.format(csv[-6:-4]))
        dfs.append(df_)
        

    df = dfs[0]
    for x in range(len(dfs)):

        df = df.append(dfs[x])

    df.columns = [x.lower() for x in df.columns]
    df.columns = [x.replace(' ','_') for x in df.columns]
    df.athlete = [x.title() for x in df.athlete]
    df = df.drop(['pos' , 'nr', 'unnamed:_7',
               'unnamed:_8', 'unnamed:_9', 'unnamed:_10'], axis=1)
    df.r1 = [x[:5] for x in df.r1]
    df.qf = [x[:5] for x in df.r1]
    df.sf = [x[:5] for x in df.sf]
    df.final = [x[:5] for x in df.final]

    df = df[['athlete', 'noc', 'games', 'r1', 'qf', 'sf', 'final']]

    df.reset_index(drop=True, inplace=True)
    value = df.final[9]
    df = df.replace(value,'0')
    df = df.fillna(value='0')
    df = df.replace("'", " ")
    df = df.replace('– (AC','0')
    df = df.replace('– (DN','0')


    df.r1 = [float(x) for x in df.r1]
    df.qf = [float(x) for x in df.qf]
    df.sf = [float(x) for x in df.sf]
    df.final = [float(x) for x in df.final]
    df = df.sort_values(by='games')
    df = df.rename(columns={'athlete':'name'})
    df = df.drop_duplicates()
    
    df.reset_index(drop=True, inplace=True)


    return df

################################################################


def create_1500m_results():
    
    '''Function to create a dataframe holding
    the results from the 1500m events
    for both men and women during the years
    2004-2016. The function fetches all the
    .csv files in the 100m_dash folder, combines
    them all into one dataframe and performs
    formatting steps on the columns and values.
    Some values in the r1, qf, sf, and final
    columns contain strings indicating if an 
    athlete was disqualified for doping or
    something that happened during the event.
    The string are removed only leaving the
    event results
    '''
    
    
    dfs = []
    csvs = glob.glob('../data/results_by_event/1500m_run/*')
    for csv in csvs:
        #print('loading_csv:{}'.format(csv))
        df_ = pd.read_csv(csv, delimiter='\t', keep_default_na=False)
        df_.insert(9, 'games', '20{}'.format(csv[-6:-4]))
        dfs.append(df_)
        

    df = dfs[0]
    for x in range(len(dfs)):

        df = df.append(dfs[x])

    df.columns = [x.lower() for x in df.columns]
    df.columns = [x.replace(' ','_') for x in df.columns]
    df.athlete = [x.title() for x in df.athlete]
    df = df.drop(['pos' , 'nr', 'unnamed:_6', 'unnamed:_7',
               'unnamed:_8', 'unnamed:_9'], axis=1)
    df.r1 = [x[:7] for x in df.r1]
    df.sf = [str(x) for x in df.sf]
    df.sf = [x[:7] for x in df.sf]
    df.final = [str(x) for x in df.final]
    df.final = [x[:7] for x in df.final]

    df = df[['athlete', 'noc', 'games', 'r1', 'sf', 'final']]

    df.reset_index(drop=True, inplace=True)
    value = df.final[12]
    df = df.replace(value,'0:00.00')
    df = df.fillna(value='0:00.00')
    df = df.replace("'", " ")
    df = df.replace('– (AC)','0:00.00')
    df = df.replace('– (DN','0:00.00')
    df = df.replace(df.sf[51], '0:00.00')
    df.r1 = df.r1.str.replace(df.r1[65][1:2], '.')
    df.sf = df.sf.str.replace(df.sf[65][1:2], '.')
    df.final = df.final.str.replace(df.final[65][1:2], '.')


    
        
    df.r1 = [(int(a) * 60 )+ int(b) + (int(c) / 1000) for a,b,c in df.r1.str.split('.')]
    df.sf = [(int(a) * 60) + int(b) + (int(c) / 1000) for a,b,c in df.sf.str.split('.')]
    df.final = [(int(a) * 60) + int(b) + (int(c) / 1000) for a,b,c in df.final.str.split('.')]


    df = df.sort_values(by='games')
    df = df.rename(columns={'athlete':'name'})
    df = df.drop_duplicates()
    
    df.reset_index(drop=True, inplace=True)


    return df

################################################################



