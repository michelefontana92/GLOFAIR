from torch.utils.data import Dataset
import os
import pandas as pd
import torch
from .utils import assign_group_id
from sklearn.utils.class_weight import compute_class_weight
import copy 
from .data_encoding import fit_scalers, encode_dataset
import pickle as pkl

pd.set_option('future.no_silent_downcasting', True)
class BaseDataset(Dataset):
    """
    BaseDataset is a custom dataset class that extends the PyTorch Dataset class. It is designed to handle datasets with 
    categorical and numerical features, sensitive attributes, and optional class weights and local weights for reweighing.
    Attributes:
        root (str): Root directory for the dataset.
        data_name (str): Name of the data file.
        data_path (str): Full path to the data file.
        target (str): Name of the target column.
        cat_cols (list): List of categorical columns.
        num_cols (list): List of numerical columns.
        sensitive_attributes (list): List of sensitive attributes.
        scaler_name (str): Name of the scaler file.
        scaler_path (str): Full path to the scaler file.
        clean_data_path (str): Path to the clean data file.
        use_class_weights (bool): Flag to use class weights.
        use_local_weights (bool): Flag to use local weights.
        id_to_combination_dict (dict): Dictionary mapping IDs to combinations of sensitive attributes.
        x (torch.Tensor): Feature tensor.
        y (torch.Tensor): Target tensor.
        positive_mask (torch.Tensor): Mask for positive samples.
        groups (dict): Dictionary of group tensors.
        groups_tensor (dict): Dictionary of group tensors.
        local_weights (dict): Dictionary of local weights tensors.
        data (pd.DataFrame): DataFrame containing the dataset.
        num_features (int): Number of features in the dataset.
        class_weights (torch.Tensor): Tensor of class weights.
        group_ids (dict): Dictionary of group IDs.
    Methods:
        setup(): Sets up the dataset by loading and preprocessing the data.
        data_preprocessing(data: pd.DataFrame): Preprocesses the data.
        _fit_scaler(): Fits the scaler to the clean data.
        _preprocess(data): Preprocesses the data using the fitted scaler.
        _load_dataset(load_data=True): Loads the dataset from the data file.
        _compute_class_weight(data: pd.DataFrame): Computes class weights for the dataset.
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns a sample from the dataset at the given index.
        get_class_weights(): Returns the class weights.
        get_group_ids(): Returns the group IDs.
        get_num_groups(group_name): Returns the number of groups for a given group name.
        get_group_cardinality(y, group_id, training_group_name): Returns the cardinality of a group.
        _compute_local_reweighing(df: pd.DataFrame, group_name: str, sensitive_dict: dict): Computes local reweighing for the dataset.
        merge(dataset): Merges the current dataset with another dataset.
    """
    def __init__(self, **kwargs):
        super(BaseDataset, self).__init__()
        self.root = kwargs.get('root', 'data')
        self.data_name = kwargs.get('data_name', 'data')
        self.data_path = os.path.join(self.root, self.data_name)
        self.target = kwargs.get('target', 'target')
        self.cat_cols = kwargs.get('cat_cols', [])
        self.num_cols = kwargs.get('num_cols', [])
        self.sensitive_attributes = kwargs.get('sensitive_attributes', [])
        self.scaler_name = kwargs.get('scaler_name', 'scaler.p')
        self.scaler_path = f'{self.root}/{self.scaler_name}'
        self.clean_data_path = ''
        self.use_class_weights = kwargs.get('use_class_weights',False)
        self.use_local_weights = kwargs.get('use_local_weights',False)
       
    def setup(self):
        self.id_to_combination_dict = {}
        x, y, groups,group_ids,local_weights = self._load_dataset()
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.positive_mask = self.y == 1
        self.groups = groups
        self.groups_tensor = {}
        self.local_weights = {}
        for group_name,group in groups.items():
            self.groups_tensor[group_name] = torch.from_numpy(group).long()
        self.group_ids = group_ids
        if self.use_local_weights:
            for group_name,_ in self.sensitive_attributes:
                self.local_weights[group_name] = torch.from_numpy(local_weights[group_name]).float()
        self.data = None
        self.num_features = self.x.shape[1]
        #print('Num groups:',self.get_num_groups('GenderRace'))
        #print('Group ids:',self.get_group_ids())
    def data_preprocessing(self,data:pd.DataFrame):
        return data
    
    def _fit_scaler(self):
        data_orig = pd.read_csv(self.clean_data_path)
        self.scaler = fit_scalers(data_orig, self.cat_cols, self.num_cols, [])
        pkl.dump(self.scaler, open(self.scaler_path, 'wb'))

    def _preprocess(self,data):

        self.scaler = pkl.load(open(self.scaler_path, 'rb'))
        data_cpy = data.copy()
        if self.use_local_weights:
            print('Computing local weights')
            for group_name,sensitive_dict in self.sensitive_attributes:
                data_cpy = self._compute_local_reweighing(data_cpy,
                                                        group_name,
                                                        sensitive_dict)
            
        for idx,label in enumerate(self.labels):
            data_cpy[self.target] = data_cpy[self.target].replace(label, idx).infer_objects()
        
        data_cpy = self.data_preprocessing(data_cpy)
        print('Assigning group ids')
        #for group_name,sensitive_dict in self.sensitive_attributes:
        data_cpy,id_to_combination_dict = assign_group_id(data_cpy, self.sensitive_attributes)
        self.id_to_combination_dict = id_to_combination_dict
        print('Encoding dataset')    
        return encode_dataset(data_cpy, self.cat_cols,
                              self.num_cols, [], 
                              self.scaler)
    

    def _load_dataset(self,load_data=True):
        if load_data:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f'File {self.data_path} not found')
            data = pd.read_csv(self.data_path)
            assert data is not None, 'Data not loaded'
        else:
            data = self.data


        print('Computing class weights')
        self.class_weights = self._compute_class_weight(data) if self.use_class_weights else None
        #print('Class weights: ',self.class_weights)

        
        try:
            print('Preprocessing data')
            data = self._preprocess(data)
        except: 
            print('Fitting scaler')
            self._fit_scaler()
            print('Preprocessing data')
            data = self._preprocess(data)
        print('Data preprocessed')
        assert data is not None, 'Data not preprocessed'
        data1 = data.copy()
        drop_cols = [self.target]
        drop_cols += [f'group_id_{group_name}' for group_name,_ in self.sensitive_attributes]
        if self.use_local_weights:
            drop_cols += [f'local_weights_{group_name}' for group_name,_ in self.sensitive_attributes]
        
        data1 = data1.drop(drop_cols, axis=1) 
        X = data1.values
        y = data[self.target].values.ravel()
        s = {}
        group_ids = {}
        weights = {}
        for group_name,_ in self.sensitive_attributes:
            s[group_name] = data[f'group_id_{group_name}'].values.ravel()
            group_ids[group_name] = torch.tensor([i for i in list(data[f'group_id_{group_name}'].unique())]).view(1,-1)
            if self.use_local_weights:
                weights[group_name] = data[f'local_weights_{group_name}'].values.ravel() 
        
        #groups_ids_list = {group_name:torch.unique(torch.tensor([i for i in range(len(group_ids[group_name]))]).view(1,-1)) for group_name,_ in self.sensitive_attributes }
        for (name,_) in self.sensitive_attributes:
            print(f"Group ids of {name}:\n {group_ids[name]}")
       
        return X, y, s,group_ids,weights
    
    def _compute_class_weight(self,data:pd.DataFrame):
        return torch.tensor(compute_class_weight('balanced',
                                                classes=data[self.target].unique(),
                                                 y=data[self.target]),
                                                dtype=torch.float32)
            
    def __len__(self):
        """__len__"""
        return len(self.x)

    def __getitem__(self, index):
        groups = {group_name:self.groups[group_name][index] for group_name,_ in self.sensitive_attributes }
        
        if self.use_local_weights:
            local_weights={group_name:self.local_weights[group_name][index] for group_name,_ in self.sensitive_attributes }
       
        
        groups_tensor = {group_name:self.groups_tensor[group_name][index] for group_name,_ in self.sensitive_attributes }
        #groups_ids_unique = {group_name:self.group_ids[group_name] for group_name,_ in self.sensitive_attributes }
        #groups_ids_list = {group_name:torch.tensor([i for i in self.group_ids[group_name]]).view(1,-1) for group_name,_ in self.sensitive_attributes }
        #print(groups_ids_list)
        result = dict(data=self.x[index], 
                    labels=self.y[index],
                    groups=groups,
                    groups_tensor = groups_tensor,
                    groups_ids_unique = self.group_ids,
                    positive_mask = self.positive_mask[index],
                    groups_ids_list = self.group_ids,
                    )
       
        if self.use_local_weights:
            result['local_weights'] = local_weights
        return result
    
    def get_class_weights(self):
        return self.class_weights
    
    def get_group_ids(self):
        return self.group_ids
    
    def get_num_groups(self,group_name):
        return len(self.group_ids[group_name][0])
    
    def get_group_cardinality(self,y,group_id,training_group_name):
        return len(torch.where((self.groups_tensor[training_group_name]==group_id) &(self.y==y))[0])
    
    def _compute_local_reweighing(self,df:pd.DataFrame,group_name:str,sensitive_dict:dict):
        data = df.copy()
        idx = f'group_id_{group_name}'
        attributes = copy.deepcopy(sensitive_dict)
        attributes.update({self.target: list(data[self.target].unique())})
        data,id_to_combination_dict = assign_group_id(data, self.sensitive_attributes)
        self.id_to_combination_dict=id_to_combination_dict
        weights = {group_id:0 for group_id in data[idx].unique()}
        for group_id in weights.keys():
            joint_proba = 1
            for attribute_name,_ in attributes.items():
                value = data[data[idx] == group_id][attribute_name].unique()[0]
               
                joint_proba *= len(data[data[attribute_name] == value]) / len(data)
            
            estimator = len(data[data[idx] == group_id]) / len(data)
            if estimator == 0 or joint_proba == 0:
                weights[group_id] = 1.0
            else:
                weights[group_id] =  joint_proba / estimator 

        data[f'local_weights_{group_name}'] = data[idx].apply(lambda x: weights[x])
       
        return data

    def merge(self,dataset):
        if self.data is None:
            data_src = pd.read_csv(self.data_path)
        else: 
            data_src = self.data
        data_dest = pd.read_csv(dataset.data_path)
        self.data = pd.concat([data_src,data_dest]).sample(frac=1).reset_index(drop=True)
        assert len(data_src.columns) == len(data_dest.columns)== len(self.data.columns), 'Columns mismatch'
        assert len(self.data) == len(data_src) + len(data_dest), 'Data mismatch'
        x, y, groups,local_weights = self._load_dataset(load_data=False)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.groups = groups
        self.local_weights = torch.from_numpy(local_weights).float()
