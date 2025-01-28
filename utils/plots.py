import pandas as pd 
import matplotlib.pyplot as plt
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def grouped_bar_plot_conditional_distribution(
                         target_attribute:str,
                         target_value:str,
                         sensitive_attributes:list,
                         data_path: str,
                         filename: str, 
                         num_clients: int,
                         start_client_idx:int,
                         save_path:str,
                         figsize: tuple = (20,20),
                         normalize:bool = True,
                         single_file=False,
                         label_name='client',
                         attributes: dict={}
                         ):
    def tuple2str(t):
        return '_'.join([str(x) for x in t])
    
    def extract_plot_data():
        indexes = []
        plot_dict = {}
        plot_dict_keys = []
        sensitive_values = [attributes[s] for s in sensitive_attributes]
        
        for name in product(*sensitive_values):
            plot_dict_keys.append(name)
            plot_dict[tuple2str(name)] = []
        print('Plot dict keys: ', plot_dict_keys)
        print('Plot dict: ', plot_dict) 
        if single_file:
          df = pd.read_csv(f'{data_path}/{filename}')
          value_counts = df[target_attribute].value_counts(normalize=normalize).to_dict()
          for key in plot_dict_keys: 
                strkey = tuple2str(key)
                try:
                    v = value_counts[key]
                except KeyError:
                    v = 0
                plot_dict[strkey].append(v)
          indexes.append(label_name)
        else:
            for i in range(num_clients):
                idx = start_client_idx+i
                df = pd.read_csv(f'{data_path}/node_{idx}/{filename}')
                for value in df[target_attribute].unique():
                    value_counts = df[(df[target_attribute]==value)]['gender'].value_counts(normalize=normalize).to_dict()
                    print(value_counts)
                    for key in plot_dict_keys: 
                        try:
                            v = value_counts[key]
                        except KeyError:
                            v = 0  
                        strkey = tuple2str(key)     
                        plot_dict[strkey].append(v)   
                    indexes.append(f'{label_name.capitalize()} {i+1}_condition_{target_value}')
        return plot_dict, indexes
    
    plt.style.use('seaborn-whitegrid')
    palette = sns.color_palette('Paired', 10)
    plot_dict, indexes = extract_plot_data()

    df = pd.DataFrame(plot_dict,index=indexes)
    #df = df.iloc[::-1]

    # Parametri per distanziare le barre
    bar_width = 0.35  # Diminuisce la larghezza delle barre
    bar_spacing = 0.7  # Aumenta lo spazio tra i gruppi di barre
    bar_spacing_within_client = 0.08
    # Posizioni per le barre
    n_groups = len(indexes)
    n_bars = len(df.columns)
    r = np.arange(n_groups) * (n_bars * (bar_width + bar_spacing_within_client) + bar_spacing)
    # Palette Paired con colori chiari per le barre con hatch
    colors = ['#87ceeb',  # Blu chiaro per rendere hatch visibile
              '#4682B4',  # Arancione medio
              '#ffb6c1',  # Verde chiaro per hatch
              '#FF69B4']  # Rosso scuro
    # Creazione delle barre con distanza
    fig, ax = plt.subplots(figsize=figsize)
    hatch_columns = ['//','xx']
    for i, col in enumerate(df.columns):
        if '<=50' in col:
            print('Hatching column: ', col)
            if 'Male' in col:
                ax.bar(r + i * (bar_width + bar_spacing_within_client), df[col], 
                   width=bar_width, label=col, color=colors[i], hatch='//', edgecolor='grey')
            else:
                ax.bar(r + i * (bar_width + bar_spacing_within_client), df[col], 
                   width=bar_width, label=col, color=colors[i], hatch='xx', edgecolor='grey')
        else:
            ax.bar(r + i * (bar_width + bar_spacing_within_client), df[col], 
                   color=colors[i],
                   width=bar_width, label=col,edgecolor='grey')

    # Migliora la leggibilità delle etichette dei client
    xticks_positions = r + (n_bars * bar_width + (n_bars - 1) * bar_spacing_within_client) / 2
    ax.set_xticks(xticks_positions)
  
    ax.set_xticklabels(indexes, rotation=45, ha='right')

    # Personalizza la griglia, legenda e etichette
    #ax.grid(True, axis='y', linestyle='--', linewidth=0.7)
    ax.legend([f'{str(x).replace("_",", income ")}' for x in df.columns],
               bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=40)
    
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    #plt.xlabel('Client', fontsize=40)
    plt.ylabel('Probability', fontsize=40)
    plt.title('Joint Distribution of Gender and Income Attributes',fontsize=40)

    plt.tight_layout()
    
    # Salva il grafico in formato PDF ad alta risoluzione
    plt.savefig(save_path, dpi=600, bbox_inches='tight', format='pdf')
    plt.close()


def grouped_bar_plot_joint_distribution(target_attributes:list,
                         data_path: str,
                         filename: str, 
                         num_clients: int,
                         start_client_idx:int,
                         save_path:str,
                         figsize: tuple = (20,20),
                         normalize:bool = True,
                         single_file=False,
                         label_name='client',
                         attributes: dict={}
                         ):
    
    """
    plt.style.use('seaborn-whitegrid')
    palette = sns.color_palette('Paired', 10)
    #plot_dict, indexes = extract_plot_data()
    #print(plot_dict)
    plot_dict_diri = {'<=50K':[0.82,0.55,0.58,0.88,0.87,0.91,0.87,0.38,0.80,0.65],
                 '>50K':[0.77,0.89,0.52,0.92,0.75,0.09,0.69,0.97,0.57,0.50]}
    plot_dict_bias = {'<=50K':[0.50,0.50,0.50,0.60,0.65,0.40,0.60,0.25,0.10,0.70],
                '>50K':[0.50,0.50,0.50,0.45,0.80,0.70,0.80,0.90,0.95,0.10]}
    indexes_diri = ['Client 1','* Client 2','Client 3','Client 4','Client 5','* Client 6','Client 7','* Client 8','Client 9','Client 10']
    indexes_bias = ['Client 1','Client 2','Client 3','Client 4','Client 5','Client 6','Client 7','* Client 8','* Client 9','* Client 10']
    df_diri = pd.DataFrame(plot_dict_diri,index=indexes_diri)
    df_bias = pd.DataFrame(plot_dict_bias,index=indexes_bias)

    colors = ['#87ceeb',  # Blu chiaro per rendere hatch visibile
              '#4682B4',  # Arancione medio
              '#ffb6c1',  # Verde chiaro per hatch
              '#FF69B4']  # Rosso scuro
    # Creazione delle barre con distanza
    # Parametri per distanziare le barre
    bar_width = 0.35  # Diminuisce la larghezza delle barre
    bar_spacing = 0.7  # Aumenta lo spazio tra i gruppi di barre
    bar_spacing_within_client = 0.08
    fontsize= 40
    fig, axes =plt.subplots(1,2,figsize=figsize)
    #fig.subplots_adjust(wspace=3) 
    ax = plt.subplot(1,2,1)
    indexes= indexes_diri
    df = df_diri 
    # Posizioni per le barre
    n_groups = len(indexes)
    n_bars = len(df.columns)
    r = np.arange(n_groups) * (n_bars * (bar_width + bar_spacing_within_client) + bar_spacing)
    # Palette Paired con colori chiari per le barre con hatch
   

    hatch_columns = ['//','xx']
    for i, col in enumerate(df.columns):
        if '<=50' in col:
            print('Hatching column: ', col)
            ax.bar(r + i * (bar_width + bar_spacing_within_client), df[col], 
                   width=bar_width, label=col, color=colors[i], hatch='//', edgecolor='grey')
           
        else:
            ax.bar(r + i * (bar_width + bar_spacing_within_client), df[col], 
                   color=colors[i],
                   width=bar_width, label=col,edgecolor='grey')

    # Migliora la leggibilità delle etichette dei client
    xticks_positions = r + (n_bars * bar_width + (n_bars - 1) * bar_spacing_within_client) / 2
    ax.set_xticks(xticks_positions)
  
    ax.set_xticklabels(indexes, rotation=45, ha='right')

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    #plt.xlabel('Client', fontsize=fontsize)
    plt.ylabel('Probability', fontsize=fontsize)
    plt.title('Dirichlet with $\\alpha=0.9$',fontsize=fontsize)
    plt.tight_layout()

    ax = plt.subplot(1,2,2)
    indexes= indexes_bias
    df = df_bias 
    # Posizioni per le barre
    n_groups = len(indexes)
    n_bars = len(df.columns)
    r = np.arange(n_groups) * (n_bars * (bar_width + bar_spacing_within_client) + bar_spacing)
    # Palette Paired con colori chiari per le barre con hatch
   

    hatch_columns = ['//','xx']
    for i, col in enumerate(df.columns):
        if '<=50' in col:
            print('Hatching column: ', col)
            ax.bar(r + i * (bar_width + bar_spacing_within_client), df[col], 
                   width=bar_width, label=col, color=colors[i], hatch='//', edgecolor='grey')
           
        else:
            ax.bar(r + i * (bar_width + bar_spacing_within_client), df[col], 
                   color=colors[i],
                   width=bar_width, label=col,edgecolor='grey')

    # Migliora la leggibilità delle etichette dei client
    xticks_positions = r + (n_bars * bar_width + (n_bars - 1) * bar_spacing_within_client) / 2
    ax.set_xticks(xticks_positions)
  
    ax.set_xticklabels(indexes, rotation=45, ha='right')

    # Personalizza la griglia, legenda e etichette
    #ax.grid(True, axis='y', linestyle='--', linewidth=0.7)
    #plt.legend([f'Pr(Male | {x})' for x in df.columns],
    #           bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=fontsize)
    plt.legend([f'Pr(Male | {x})' for x in df.columns],
              loc='best', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    #plt.xlabel('Client', fontsize=fontsize)
    #plt.ylabel('Probability', fontsize=fontsize)
    
    plt.title('Bias-Controlled',fontsize=fontsize)
    plt.suptitle('Adult Conditional Distribution',fontsize=fontsize)
    plt.tight_layout()
    
    # Salva il grafico in formato PDF ad alta risoluzione
    plt.savefig(save_path, dpi=600, bbox_inches='tight', format='pdf')
    #plt.savefig(save_path,bbox_inches='tight', format='jpg')
    plt.close()
    """
    # Stile del grafico
    plt.style.use('seaborn-whitegrid')
   
    # Dati
    plot_dict_diri = {'<=50K': [0.82, 0.55, 0.58, 0.88, 0.87, 0.91, 0.87, 0.38, 0.80, 0.65],
                    '>50K': [0.77, 0.89, 0.52, 0.92, 0.75, 0.09, 0.69, 0.97, 0.57, 0.50]}
    plot_dict_bias = {'<=50K': [0.50, 0.50, 0.50, 0.60, 0.65, 0.40, 0.60, 0.25, 0.10, 0.70],
                    '>50K': [0.50, 0.50, 0.50, 0.45, 0.80, 0.70, 0.80, 0.90, 0.95, 0.10]}
    indexes_diri = ['Client 1', '* Client 2', 'Client 3', 'Client 4', 'Client 5', '* Client 6', 'Client 7', '* Client 8', 'Client 9', 'Client 10']
    indexes_bias = ['Client 1', 'Client 2', 'Client 3', 'Client 4', 'Client 5', 'Client 6', 'Client 7', '* Client 8', '* Client 9', '* Client 10']

    df_diri = pd.DataFrame(plot_dict_diri, index=indexes_diri)
    df_bias = pd.DataFrame(plot_dict_bias, index=indexes_bias)
    hatch_columns = ['//','xx']
    # Parametri grafici
    colors = ['#87ceeb', '#4682B4', '#ffb6c1', '#FF69B4']
    bar_width = 0.35
    bar_spacing = 0.7
    bar_spacing_within_client = 0.08
    fontsize = 40  # Ridotto per visualizzazioni normali
   

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.4)  # Aggiungi spazio tra i subplot

    # Subplot 1
    ax = axes[0]
    indexes = indexes_diri
    df = df_diri
    n_groups = len(indexes)
    n_bars = len(df.columns)
    r = np.arange(n_groups) * (n_bars * (bar_width + bar_spacing_within_client) + bar_spacing)

    for i, col in enumerate(df.columns):
        ax.bar(r + i * (bar_width + bar_spacing_within_client), df[col], 
            width=bar_width, label=col, color=colors[i], edgecolor='grey', 
            hatch='//')

    ax.set_xticks(r + (n_bars * bar_width + (n_bars - 1) * bar_spacing_within_client) / 2)
    ax.set_xticklabels(indexes, rotation=45, ha='right')
    ax.set_title('Dirichlet with $\\alpha=0.9$', fontsize=fontsize)
    ax.set_ylabel('Probability', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    # Subplot 2
    ax = axes[1]
    indexes = indexes_bias
    df = df_bias
    n_groups = len(indexes)
    n_bars = len(df.columns)
    r = np.arange(n_groups) * (n_bars * (bar_width + bar_spacing_within_client) + bar_spacing)

    for i, col in enumerate(df.columns):
        ax.bar(r + i * (bar_width + bar_spacing_within_client), df[col], 
            width=bar_width, label=col, color=colors[i], edgecolor='grey',
            hatch='//')

    ax.set_xticks(r + (n_bars * bar_width + (n_bars - 1) * bar_spacing_within_client) / 2)
    ax.set_xticklabels(indexes, rotation=45, ha='right')
    ax.set_title('Bias-Controlled', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    # Titolo generale
    plt.suptitle('Adult Conditional Distribution of Gender given Income', fontsize=fontsize + 4)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Legenda
    plt.legend([f'Pr(Male | {x})' for x in df.columns], loc='upper left', bbox_to_anchor=(-0.015,1.03), fontsize=fontsize-5, ncol=1)

    # Salvataggio del grafico
    plt.savefig(save_path, dpi=600, bbox_inches='tight', format='pdf')
    plt.close()

def horizontal_stack_plot(target_attributes:list,
                         data_path: str,
                         filename: str, 
                         num_clients: int,
                         start_client_idx:int,
                         save_path:str,
                         figsize: tuple = (20,20),
                         normalize:bool = True,
                         single_file=False,
                         label_name='client',
                         attributes: dict={}
                         ):
    def tuple2str(t):
        return '_'.join([str(x) for x in t])
    
    def extract_plot_data():
        
        indexes = []
        plot_dict = {}
        plot_dict_keys = []
        sensitive_values = [attributes[s] for s in target_attributes]
        
        for name in product(*sensitive_values):
            #print('Key name: ', name)
            plot_dict_keys.append(name)
            plot_dict[tuple2str(name)] = []
            
        if single_file:
          df = pd.read_csv(f'{data_path}/{filename}')
          value_counts = df[target_attributes].value_counts(normalize=normalize).to_dict()
          print(value_counts.keys())
          for key in plot_dict_keys: 
                strkey = tuple2str(key)
                try:
                    v = value_counts[key]
                except KeyError:
                    v = 0
                plot_dict[strkey].append(v)
          indexes.append(label_name)
        else:
            for i in range(num_clients):
                idx = start_client_idx+i
                df = pd.read_csv(f'{data_path}/node_{idx}/{filename}')
                value_counts = df[target_attributes].value_counts(normalize=normalize).to_dict()
                #print(value_counts.keys())
                for key in plot_dict_keys: 
                    try:
                        v = value_counts[key]
                    except KeyError:
                        v = 0  
                    strkey = tuple2str(key)     
                    plot_dict[strkey].append(v)   
                indexes.append(f'{label_name}_{i+1}')
        return plot_dict, indexes

    plot_dict, indexes = extract_plot_data()

    df = pd.DataFrame(plot_dict,index=indexes)
    df = df.iloc[::-1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    # create a stacked barplot rotated by 90 degrees
    ax = df.plot(kind='barh', stacked=True,
            rot=15,
            figsize=figsize,
            color=colors)
    
    hatch_patterns = ['//', '\\\\', 'xx', '..']
    bars = ax.patches  # Ottieni tutte le barre create dal grafico
    for i, bar in enumerate(bars):
        bar.set_hatch(hatch_patterns[i // 10])  # Applica pattern ciclicamente
    #ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)
    # set the yticks font size
    plt.yticks(fontsize=50)
    plt.xticks(fontsize=50)
    plt.xlabel('Probability ',fontsize=50)
    plt.legend([f'{str(x).replace("_",", income ")}' for x in df.columns],
     bbox_to_anchor=(1.02, 1), loc=2,fontsize=45)
    # replace the _ with a " with " in the legend
    plt.title('Income Distribution by Gender and Income Category',fontsize=45)
    plt.tight_layout()
    
    plt.savefig(save_path,dpi=600,bbox_inches='tight',format='pdf')
    plt.close()


def horizontal_stack_plot_size( 
                         data_path: str,
                         filename: str, 
                         num_clients: int,
                         start_client_idx:int,
                         save_path:str,
                         figsize: tuple = (20,20),
                         single_file=False,
                         label_name='client'):
   
    def extract_plot_data():
        indexes = []
        strkey = 'dataset size'
        plot_dict = {strkey:[]}
        if single_file:
            total_len = 0
            for file in filename:
                df = pd.read_csv(f'{data_path}/node_{idx}/{file}')
                total_len += len(df)
            plot_dict[strkey].append(total_len)
            indexes.append(label_name)
        else:
            for i in range(num_clients):
                idx = start_client_idx+i
                total_len = 0
                for file in filename:
                    df = pd.read_csv(f'{data_path}/node_{idx}/{file}')
                    total_len += len(df)
                plot_dict[strkey].append(total_len)
                indexes.append(f'{label_name}_{i+1}')
        return plot_dict, indexes

    plot_dict, indexes = extract_plot_data()

    df = pd.DataFrame(plot_dict,index=indexes)
    
    df.plot(kind='bar', stacked=True, rot=15,figsize=figsize)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2)
    plt.ylabel('Dataset size')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()