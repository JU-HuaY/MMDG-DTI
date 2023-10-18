import json
import numpy as np

# data_fold = json.load(open("davis_div.txt"))
dir_input = "Davis/data_split/"
sequences = np.load(dir_input + 'sequences.npy',allow_pickle=True)
smiles = np.load(dir_input + 'smiles.npy', allow_pickle=True)
molecule_words = np.load(dir_input + 'molecule_words.npy',allow_pickle=True)
molecule_atoms = np.load(dir_input + 'molecule_atoms.npy',allow_pickle=True)
molecule_adjs = np.load(dir_input + 'molecule_adjs.npy',allow_pickle=True)
proteins = np.load(dir_input + 'proteins.npy',allow_pickle=True)
affinity = np.load(dir_input + 'interactions.npy',allow_pickle=True)
print(len(smiles))

with open(dir_input + "train.txt", "r") as f:  # 打开文件
    train_list = f.read()
    train_list = train_list.split(', ')
    train_list = np.array(train_list,dtype=np.int)

with open(dir_input + "test.txt", "r") as f:  # 打开文件
    test_list = f.read()
    test_list = test_list.split(', ')
    test_list = np.array(test_list,dtype=np.int)

# train_fold=np.loadtxt(dir_input + 'train.txt', dtype=np.int)
print(len(test_list))
molecule_words_train, molecule_atoms_train, molecule_adjs_train, proteins_train, affinity_train = [], [], [], [], []
molecule_words_test, molecule_atoms_test, molecule_adjs_test, proteins_test, affinity_test = [], [], [], [], []
sequences_train, smiles_train = [], []
sequences_test, smiles_test = [], []


for j in range(len(train_list)):
    molecule_words_train.append(np.array(molecule_words[train_list[j]]))
    molecule_atoms_train.append(molecule_atoms[train_list[j]])
    molecule_adjs_train.append(molecule_adjs[train_list[j]])
    proteins_train.append(proteins[train_list[j]])
    affinity_train.append(affinity[train_list[j]])
    sequences_train.append(sequences[train_list[j]])
    smiles_train.append(smiles[train_list[j]])

for j in range(len(test_list)):
    molecule_words_test.append(np.array(molecule_words[test_list[j]]))
    molecule_atoms_test.append(molecule_atoms[test_list[j]])
    molecule_adjs_test.append(molecule_adjs[test_list[j]])
    proteins_test.append(proteins[test_list[j]])
    affinity_test.append(affinity[test_list[j]])
    sequences_test.append(sequences[test_list[j]])
    smiles_test.append(smiles[test_list[j]])


np.save("Davis/train/" + 'molecule_words', molecule_words_train)
np.save("Davis/train/" + 'molecule_atoms', molecule_atoms_train)
np.save("Davis/train/" + 'molecule_adjs', molecule_adjs_train)
np.save("Davis/train/" + 'proteins', proteins_train)
np.save("Davis/train/" + 'interactions', affinity_train)
np.save("Davis/train/" + 'sequences', sequences_train)
np.save("Davis/train/" + 'smiles', smiles_train)

np.save("Davis/test/" + 'molecule_words', molecule_words_test)
np.save("Davis/test/" + 'molecule_atoms', molecule_atoms_test)
np.save("Davis/test/" + 'molecule_adjs', molecule_adjs_test)
np.save("Davis/test/" + 'proteins', proteins_test)
np.save("Davis/test/" + 'interactions', affinity_test)
np.save("Davis/test/" + 'sequences', sequences_test)
np.save("Davis/test/" + 'smiles', smiles_test)

