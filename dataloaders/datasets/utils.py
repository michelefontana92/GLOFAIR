import numpy as np
def find_bucket(n,thresholds):
    for i in range(len(thresholds)-1):
        if n>=thresholds[i] and n<=thresholds[i+1]:
            return i
    if n>thresholds[-1]:
        return len(thresholds)-1
    return -1

def _assign(x,sensitive_dict,key):
    
    if x in sensitive_dict[key]:
        return x
    else:
        #print(f'Assigned NoneGroup to {x}')
        return 'NoneGroup'


import itertools

def assign_group_id(df, sensitive_attributes):
    """
    Assegna un identificatore di gruppo a ciascuna riga di un DataFrame in base a combinazioni di colonne specificate.
    
    Parametri:
    df (pd.DataFrame): DataFrame di input con colonne da combinare.
    group_dict (dict): Dizionario che contiene le colonne e i valori possibili da combinare.
    
    Ritorna:
    tuple: Una tupla contenente:
        - pd.DataFrame: DataFrame con una nuova colonna 'GroupID'.
        - dict: Dizionario che mappa gli ID di gruppo alle combinazioni di attributi.
    """
    id_to_combination={}
    for (name,group_dict) in sensitive_attributes:
        # Ottieni i nomi delle colonne e i valori possibili per ciascuna colonna
        columns = list(group_dict.keys())
        values = [group_dict[col] for col in columns]

        # Genera tutte le combinazioni possibili e assegna un ID univoco
        combinations = list(itertools.product(*values))
        combination_ids = {tuple(combination): idx for idx, combination in enumerate(combinations)}
        
        # Creiamo il dizionario id: combinazione
        id_to_combination[name] = {idx: dict(zip(columns, combination)) for combination, idx in combination_ids.items()}

        # Funzione per ottenere l'ID della combinazione per ogni riga del DataFrame
        def get_combination_id(row):
            row_tuple = tuple(row[col] for col in columns)
            return combination_ids.get(row_tuple, -1)

        # Creiamo la nuova colonna 'GroupID' nel DataFrame
        df[f'group_id_{name}'] = df.apply(get_combination_id, axis=1)
    return df, id_to_combination


def assign_group_id_old(data,sensitive_dict,group_name):
    if len(sensitive_dict.keys()) == 0:
        data[f'group_id_{group_name}'] = data.apply(lambda x: -1, axis=1)
        return data
    for key in sensitive_dict.keys():
        if type(data[key].iloc[0]) == str:
            #print(f'Assigning group id for {key}')
            #print('Sensitive dict:',sensitive_dict[key])
            data[key+'_new'] = data[key].apply(lambda x:_assign(x,sensitive_dict,key))#lambda x: x if x in sensitive_dict[key] else 'NoneGroup')
        elif str(data[key].iloc[0]).isnumeric():
            thresholds = sensitive_dict[key]
            #sort the thresholds
            if np.array(thresholds).min() > data[key].min():
                thresholds.append(data[key].min())
            if np.array(thresholds).max() < data[key].max():
                thresholds.append(data[key].max())
            
            thresholds.sort()
           
            data[key+'_new'] = data[key].apply(lambda x: find_bucket(x,thresholds))
            data[key+'_new'] = data[key+'_new'].apply(str)
        else:
            raise ValueError('Data type not supported')
        
    sensitive = [x+'_new' for x in sensitive_dict.keys()]
    data['group']= data[sensitive].apply(lambda x: ''.join(x), axis=1)
    group_ids = {}
    current_id = 0
    for group in data['group'].unique():
        if 'NoneGroup' in group:
            group_ids[group] = -1
        else:
            group_ids[group] = current_id
            current_id += 1
    data[f'group_id_{group_name}'] = data['group'].map(group_ids)
    data.drop(sensitive+['group'],axis=1,inplace=True)
    return data