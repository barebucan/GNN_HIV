
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import os.path as osp
import torch
from tqdm import tqdm 
import os
import numpy as np
from torch_geometric.data import Dataset, download_url, Data


class MyOwnDataset(Dataset):
    def __init__(self, root, filename ,test = False, transform=None, pre_transform=None, pre_filter=None):
        
        self.test = test
        self.filename = filename
        super().__init__(root, transform, pre_transform, pre_filter)
   
    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
       pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, mol in tqdm(self.data.iterrows(),total = self.data.shape[0]):
            
            mol_obj = Chem.MolFromSmiles(mol["smiles"])

            nodes = self.get_nodes(mol_obj)

            edge_index = self._get_adjacency_info(mol_obj)

            edge = self.get_edge(mol_obj)

            label = mol["HIV_active"]

            data = Data(x = nodes,
                        edge_attr= edge,
                        edge_index= edge_index,
                        y= label
                        )
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f"data_test_{index}.pt"))
            else:
                torch.save(data, os.path.join(self.processed_dir, f"data_{index}.pt"))

    def get_nodes(self, mol):

        all_atoms = []

        for atom in mol.GetAtoms():
            node = []
            node.append(atom.GetAtomicNum())
            node.append(atom.GetDegree())
            node.append(atom.GetFormalCharge())
            node.append(atom.GetHybridization())
            node.append(atom.GetIsAromatic())

            all_atoms.append(node)
        
        return torch.tensor(np.asarray(all_atoms), dtype = torch.float)
    
    def get_edge(self, mol):

        all_egdes = []

        for bond in mol.GetBonds():
            edge = []
            edge.append(bond.GetBondTypeAsDouble())
            edge.append(bond.IsInRing())
            edge.append(bond.GetIsAromatic())
            all_egdes.append(edge)

        return torch.tensor(np.asarray(all_egdes), dtype = torch.float)
    
    def _get_adjacency_info(self, mol):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])

        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t()  # Transpose to get the desired shape
        return edge_indices

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))
            return data
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
            return data