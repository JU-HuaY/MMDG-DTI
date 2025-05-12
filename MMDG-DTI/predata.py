# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:06:40 2020

@author: 华阳
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from graph_features import atom_features
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
import torch
import pickle

BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)


protein_dict = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                 "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                 "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                 "U": 19, "T": 20, "W": 21,
                 "V": 22, "Y": 23, "X": 24,
                 "Z": 25 }

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

num_atom_feat = 34

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]

    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in atoms])

    n_edge_features = 4
    edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
        edge_features[i, j, bond_type] = 1
        edge_features[j, i, bond_type] = 1

    return node_features, adjacency

def first_sequence(sequence):
    words = [protein_dict[sequence[i]]
             for i in range(len(sequence))]
    return np.array(words)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')
# 
prot_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
prot_model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd").to(device)

chem_tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k", do_lower_case=False)
chem_model = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k").to(device)

def DTI_datasets(dataset, dir_input):  # BindingDB, Human, C.elegan, GPCRs
    with open(dataset,"r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    molecule_atoms, molecule_adjs, molecule_words, proteins, sequencess, smiless, interactions = [], [], [], [], [], [], []
    p_LM, d_LM = {}, {}
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))

        smiles, sequences, interaction = data.strip().split(" ")
        if len(sequences) > 5000:
            sequences = sequences[0:5000]
        sequencess.append(sequences)
        smiless.append(smiles)
        # print(len(sequences))
        protein_input = prot_tokenizer.batch_encode_plus([" ".join(sequences)], add_special_tokens=True, padding=True)#"longest", max_length=1200, truncation=True, return_tensors='pt')
        p_IDS = torch.tensor(protein_input["input_ids"]).to(device)
        p_a_m = torch.tensor(protein_input["attention_mask"]).to(device)
        # sequences = torch.tensor(sequences).to(self.device)
        with torch.no_grad():
            prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)
        prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()
        if sequences not in p_LM:
            p_LM[sequences] = prot_feature

        chem_input = chem_tokenizer.batch_encode_plus([smiles], add_special_tokens=True, padding=True)
        c_IDS = torch.tensor(chem_input["input_ids"]).to(device)
        c_a_m = torch.tensor(chem_input["attention_mask"]).to(device)
        with torch.no_grad():
            chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)
        chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
        if smiles not in d_LM:
            d_LM[smiles] = chem_feature

        molecule_word = []
        for i in range(len(smiles)):
            molecule_word.append(CHAR_SMI_SET[smiles[i]])
        molecule_word = np.array(molecule_word)
        molecule_words.append(molecule_word)

        atom_feature, adj = smile_to_graph(smiles)
        molecule_atoms.append(atom_feature)
        molecule_adjs.append(adj)

        protein_first = first_sequence(sequences)
        proteins.append(protein_first)

        interactions.append(np.array([float(interaction)]))

    with open(dir_input + "p_LM.pkl", "wb") as p:
        pickle.dump(p_LM, p)

    with open(dir_input + "d_LM.pkl", "wb") as d:
        pickle.dump(d_LM, d)

    np.save(dir_input + 'molecule_words', molecule_words)
    np.save(dir_input + 'molecule_atoms', molecule_atoms)
    np.save(dir_input + 'molecule_adjs', molecule_adjs)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'sequences', sequencess)
    np.save(dir_input + 'smiles', smiless)
    np.save(dir_input + 'interactions', interactions)

def DTI_drugbank_datasets(dataset, dir_input):  # DrugBank
    with open(dataset,"r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    molecule_atoms, molecule_adjs, molecule_words, proteins, sequencess, smiless, interactions = [], [], [], [], [], [], []
    p_LM, d_LM = {}, {}
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))

        _, _, smiles, sequences, interaction = data.strip().split(" ")
        if len(sequences) > 5000:
            sequences = sequences[0:5000]
        # if len(smiles) > 512:
        #     smiles = smiles[0:512]
        sequencess.append(sequences)
        smiless.append(smiles)
        # print(len(sequences))
        protein_input = prot_tokenizer.batch_encode_plus([" ".join(sequences)], add_special_tokens=True, padding=True)#"longest", max_length=1200, truncation=True, return_tensors='pt')
        p_IDS = torch.tensor(protein_input["input_ids"]).to(device)
        p_a_m = torch.tensor(protein_input["attention_mask"]).to(device)
        # sequences = torch.tensor(sequences).to(self.device)
        with torch.no_grad():
            prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)
        prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()
        if sequences not in p_LM:
            p_LM[sequences] = prot_feature
        print(len(smiles))
        if len(smiles) < 512:
            chem_input = chem_tokenizer.batch_encode_plus([smiles], add_special_tokens=True, padding=True)
            c_IDS = torch.tensor(chem_input["input_ids"]).to(device)
            c_a_m = torch.tensor(chem_input["attention_mask"]).to(device)
            with torch.no_grad():
                chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)
            chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
            if smiles not in d_LM:
                d_LM[smiles] = chem_feature
        else:
            smiles_short = smiles[0:512]
            chem_input = chem_tokenizer.batch_encode_plus([smiles_short], add_special_tokens=True, padding=True)
            c_IDS = torch.tensor(chem_input["input_ids"]).to(device)
            c_a_m = torch.tensor(chem_input["attention_mask"]).to(device)
            with torch.no_grad():
                chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)
            chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
            if smiles not in d_LM:
                d_LM[smiles] = chem_feature

        molecule_word = []
        for i in range(len(smiles)):
            molecule_word.append(CHAR_SMI_SET[smiles[i]])
        molecule_word = np.array(molecule_word)
        molecule_words.append(molecule_word)

        atom_feature, adj = smile_to_graph(smiles)
        molecule_atoms.append(atom_feature)
        molecule_adjs.append(adj)

        protein_first = first_sequence(sequences)
        proteins.append(protein_first)

        interactions.append(np.array([float(interaction)]))

    with open(dir_input + "p_LM.pkl", "wb") as p:
        pickle.dump(p_LM, p)

    with open(dir_input + "d_LM.pkl", "wb") as d:
        pickle.dump(d_LM, d)

    np.save(dir_input + 'molecule_words', molecule_words)
    np.save(dir_input + 'molecule_atoms', molecule_atoms)
    np.save(dir_input + 'molecule_adjs', molecule_adjs)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'sequences', sequencess)
    np.save(dir_input + 'smiles', smiless)
    np.save(dir_input + 'interactions', interactions)

def DTA_datasets(dataset, dir_input): # Davis, Kiba
    with open(dataset,"r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    molecule_atoms, molecule_adjs, molecule_words, proteins, affinities = [], [], [], [], []
    p_LM, d_LM = {}, {}
    p_sequences, d_smiles = [], []
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))

        _, _, smiles, sequences, affinity = data.strip().split(" ")
        p_sequences.append(sequences)
        d_smiles.append(smiles)
        protein_input = prot_tokenizer.batch_encode_plus([" ".join(sequences)], add_special_tokens=True,
                                                         padding=True)  # "longest", max_length=1200, truncation=True, return_tensors='pt')
        p_IDS = torch.tensor(protein_input["input_ids"]).to(device)
        p_a_m = torch.tensor(protein_input["attention_mask"]).to(device)
        # sequences = torch.tensor(sequences).to(self.device)
        with torch.no_grad():
            prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)
        prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()
        if sequences not in p_LM:
            p_LM[sequences] = prot_feature
        print(len(smiles))
        if len(smiles) < 512:
            chem_input = chem_tokenizer.batch_encode_plus([smiles], add_special_tokens=True, padding=True)
            c_IDS = torch.tensor(chem_input["input_ids"]).to(device)
            c_a_m = torch.tensor(chem_input["attention_mask"]).to(device)
            with torch.no_grad():
                chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)
            chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
            if smiles not in d_LM:
                d_LM[smiles] = chem_feature
        else:
            smiles_short = smiles[0:512]
            chem_input = chem_tokenizer.batch_encode_plus([smiles_short], add_special_tokens=True, padding=True)
            c_IDS = torch.tensor(chem_input["input_ids"]).to(device)
            c_a_m = torch.tensor(chem_input["attention_mask"]).to(device)
            with torch.no_grad():
                chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)
            chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
            if smiles not in d_LM:
                d_LM[smiles] = chem_feature

        molecule_word = []
        for i in range(len(smiles)):
            molecule_word.append(CHAR_SMI_SET[smiles[i]])
        molecule_word = np.array(molecule_word)
        molecule_words.append(molecule_word)

        atom_feature, adj = smile_to_graph(smiles)
        molecule_atoms.append(atom_feature)
        molecule_adjs.append(adj)

        protein_first = first_sequence(sequences)
        proteins.append(protein_first)
        affinities.append(np.array([float(affinity)]))

    with open(dir_input + "p_LM.pkl", "wb") as p:
        pickle.dump(p_LM, p)

    with open(dir_input + "d_LM.pkl", "wb") as d:
        pickle.dump(d_LM, d)

    np.save(dir_input + 'sequences', p_sequences)
    np.save(dir_input + 'smiles', d_smiles)
    np.save(dir_input + 'molecule_words', molecule_words)
    np.save(dir_input + 'molecule_atoms', molecule_atoms)
    np.save(dir_input + 'molecule_adjs', molecule_adjs)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'affinity', affinities)
    
def DTI_datasets3(dataset, dir_input):  # BindingDB, Human, C.elegan, GPCRs
    with open(dataset,"r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    molecule_atoms, molecule_adjs, molecule_words, proteins, sequencess, smiless, interactions = [], [], [], [], [], [], []
    # p_LM, d_LM = {}, {}
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))

        smiles, sequences, interaction = data.strip().split(" ")
        if len(sequences) > 5000:
            sequences = sequences[0:5000]
        sequencess.append(sequences)
        smiless.append(smiles)
        # print(len(sequences))
        # protein_input = prot_tokenizer.batch_encode_plus([" ".join(sequences)], add_special_tokens=True, padding=True)#"longest", max_length=1200, truncation=True, return_tensors='pt')
        # p_IDS = torch.tensor(protein_input["input_ids"]).to(device)
        # p_a_m = torch.tensor(protein_input["attention_mask"]).to(device)
        # # sequences = torch.tensor(sequences).to(self.device)
        # with torch.no_grad():
        #     prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)
        # prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()
        # if sequences not in p_LM:
        #     p_LM[sequences] = prot_feature

        # chem_input = chem_tokenizer.batch_encode_plus([smiles], add_special_tokens=True, padding=True)
        # c_IDS = torch.tensor(chem_input["input_ids"]).to(device)
        # c_a_m = torch.tensor(chem_input["attention_mask"]).to(device)
        # with torch.no_grad():
        #     chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)
        # chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
        # if smiles not in d_LM:
        #     d_LM[smiles] = chem_feature

        molecule_word = []
        for i in range(len(smiles)):
            molecule_word.append(CHAR_SMI_SET[smiles[i]])
        molecule_word = np.array(molecule_word)
        molecule_words.append(molecule_word)

        atom_feature, adj = smile_to_graph(smiles)
        molecule_atoms.append(atom_feature)
        molecule_adjs.append(adj)

        protein_first = first_sequence(sequences)
        proteins.append(protein_first)

        interactions.append(np.array([float(interaction)]))

    # with open(dir_input + "p_LM.pkl", "wb") as p:
    #     pickle.dump(p_LM, p)
    #
    # with open(dir_input + "d_LM.pkl", "wb") as d:
    #     pickle.dump(d_LM, d)

    np.save(dir_input + 'molecule_words', molecule_words)
    np.save(dir_input + 'molecule_atoms', molecule_atoms)
    np.save(dir_input + 'molecule_adjs', molecule_adjs)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'sequences', sequencess)
    np.save(dir_input + 'smiles', smiless)
    np.save(dir_input + 'interactions', interactions)


if __name__ == "__main__":

    DTI_datasets3("BindingDB_cold/unseen_pair_setting/Unseen pair/train.txt", 'BindingDB_cold/unseen_pair_setting/train/')
    DTI_datasets3("BindingDB_cold/unseen_pair_setting/Unseen pair/test.txt", 'BindingDB_cold/unseen_pair_setting/test/')
    # #
    # DTI_datasets("GPCRs/original/GPCR_train.txt", 'GPCRs/train/')
    # DTI_datasets("GPCRs/original/GPCR_test.txt", 'GPCRs/test/')
    #
    # DTI_drugbank_datasets("DrugBank/original/data.txt", 'DrugBank/data_split/')
    # DTI_datasets("Human/original/data.txt", 'Human/data_split/')
    # DTI_datasets("C.elegans/original/data.txt", 'C.elegans/data_split/')

    # DTI_drugbank_datasets("Kiba/original/KIBA_DTI.txt", 'Kiba/data_split/')
    # DTI_drugbank_datasets("Davis/original/Davis_DTI.txt", 'Davis/data_split/')

    print('The preprocess of dataset has finished!')
