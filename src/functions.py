import pandas as pd



'''This file holds the functions that I will be using
for the project. You can find the docstrings for each
function below the line of code defining the function.'''


################################################################

def content_cleaner(df):
    df = df.dropna(axis=0)
    df['rank'] = [x.strip('\n') for x in df['rank']]
    df['rank'] = [x.replace('G', '1.') for x in df['rank']]
    df['rank'] = [x.replace('S', '2.') for x in df['rank']]
    df['rank'] = [x.replace('B', '3.') for x in df['rank']]
    
    df['name'] = [x.strip('\n\n\n\n\n\n\n\n\n\n\n\n') for x in df['name']]
    df['name'] = [x.replace('\n', ' ') for x in df['name']]
    df['name'] = [x[:-3] for x in df['name']]
    df['name'] = [x.title() for x in df['name']]   
    
    df['result'] = [x[:-4] for x in df['result']]
    df['result'] = [x.split('\r\n') for x in df['result']]
    df['result'] = [x[1] for x in df['result']]
    df['result'] = [x.replace('h', ':') for x in df['result']]
    
    return df


################################################################

def athens_scraper(content_):
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