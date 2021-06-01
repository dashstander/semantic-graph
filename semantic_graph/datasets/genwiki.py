from csv import DictWriter
import hyperjson as json
import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List


class GenwikiEntities:
    scopes = ['fine', 'full']
    modes = ['train', 'test']

    def __init__(self):
        self.file_dir = os.getcwd() + '/genwiki'
        self.entities = {}
        self.relationships = {}
        self.r_idx = 0
        self.e_idx = 0
        self.triples = set()

    def update_relationships(self, rels: List[str]):
        for r in rels:
            if r not in self.relationships:
                self.relationships.update({r: self.r_idx})
                self.r_idx += 1
        
    def update_entities(self, ents: List[str]):
        for e in ents:
            if e not in self.entities:
                self.entities.update({e: self.e_idx})
                self.e_idx += 1

    def _load_data(self, scope, mode):
        data = []
        for file in os.scandir(f'{self.file_dir}/{mode}/{scope}'):
            data += json.load(file)
        return data

    def triple_tensor(self, triple: List[str]):
        h_idx = self.entities[triple[0]]
        r_idx = self.relationships[triple[1]]
        t_idx = self.entities[triple[2]]
        return torch.tensor([h_idx, r_idx, t_idx], dtype=torch.int32)

    def write_entity_csv(self):
        with open(f'{self.file_dir}/entity.csv', mode='w') as csv_file:
            writer = DictWriter(csv_file, fieldnames=['entity', 'id'])
            writer.writeheader()
            writer.writerows([{'entity': e, 'id': i} for e, i in self.entities.items()])
    
    def write_relationship_csv(self):
        with open(f'{self.file_dir}/relationship.csv', mode='w') as csv_file:
            writer = DictWriter(csv_file, fieldnames=['relationship', 'id'])
            writer.writeheader()
            writer.writerows([{'relationship': e, 'id': i} for e, i in self.relationships.items()])

    def write_triple_tensors(self):
        with open(f'{self.file_dir}/triples.p') as pkl_file:
            pickle.dump(self.triples, pkl_file)

    def process_data(self):
        for mode in self.modes:
            for scope in self.scopes:
                for data in self._load_data(scope, mode):
                    for triple in data['graph']:
                        self.update_relationships([triple[1]])
                        self.update_entities([triple[0], triple[2]])
                        self.triples.add(self.triple_tensor(triple))
        self.write_entity_csv()
        self.write_relationship_csv()
        self.write_triple_tensors()
        




    

    

class GenwikiTriples(Dataset):

    def __init__(self, mode, scope):
        super().__init__()
        self.file_dir = os.getcwd() + '/genwiki/{mode}/{scope}'
        self.entities = set()
        self.relationships = set()  
        self.e_idx = 0
        self.rel_idx = 0
        self.triples = []
    
    def _load_data(self):
        triples = []
        for file in os.scandir(self.file_dir):
            data = json.load(file)
            self.entities.add()
        return data