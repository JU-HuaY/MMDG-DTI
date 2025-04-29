import scipy.sparse as sp
import pickle
import sys
import timeit
import scipy
import numpy as np
from math import sqrt
import scipy
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from data_merge import data_load
from networks.model import DGMM_DTI
import warnings

# warnings.filterwarnings("ignore")
# from transformers import AutoModel, AutoTokenizer, pipelines
import os
from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.distributed import DistributedSampler

torch.multiprocessing.set_start_method('spawn')

def pack(molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, p_LMs, d_LMs, device, sources=None):

    proteins_len = 1200
    words_len = 100
    atoms_len = 0
    p_l = 1200
    d_l = 100
    N = len(molecule_atoms)
    molecule_words_new = torch.zeros((N, words_len), device=device)
    i = 0
    for molecule_word in molecule_words:
        molecule_word_len = molecule_word.shape[0]
        # print(compounds_word.shape)
        if molecule_word_len <= 100:
            molecule_words_new[i, :molecule_word_len] = molecule_word
        else:
            molecule_words_new[i] = molecule_word[0:100]
        i += 1

    atom_num = []
    for atom in molecule_atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    molecule_atoms_new = torch.zeros((N, atoms_len, 75), device=device)
    i = 0
    for atom in molecule_atoms:
        a_len = atom.shape[0]
        molecule_atoms_new[i, :a_len, :] = atom
        i += 1

    molecule_adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in molecule_adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len, device=device)
        molecule_adjs_new[i, :a_len, :a_len] = adj
        i += 1

    proteins_new = torch.zeros((N, proteins_len), device=device)
    i = 0
    for protein in proteins:
        if protein.shape[0] > 1200:
            protein = protein[0:1200]
        a_len = protein.shape[0]
        proteins_new[i, :a_len] = protein
        i += 1

    protein_LMs = []
    molecule_LMs = []
    for sequence in sequences:
        protein_LMs.append(p_LMs[sequence])

    for smile in smiles:
        molecule_LMs.append(d_LMs[smile])
        # if d_LMs[smile].shape[0] > d_l:
        #     d_l = d_LMs[smile].shape[0]

    protein_LM = torch.zeros((N, p_l, 1024), device=device)
    molecule_LM = torch.zeros((N, d_l, 768), device=device)
    # print(d_l)
    for i in range(N):
        C_L = molecule_LMs[i].shape[0]
        if C_L >= 100:
            molecule_LM[i, :, :] = torch.tensor(molecule_LMs[i][0:100, :]).to(device)
        else:
            molecule_LM[i, :C_L, :] = torch.tensor(molecule_LMs[i]).to(device)
        P_L = protein_LMs[i].shape[0]
        if P_L >= 1200:
            protein_LM[i, :, :] = torch.tensor(protein_LMs[i][0:1200, :]).to(device)
        else:
            protein_LM[i, :P_L, :] = torch.tensor(protein_LMs[i]).to(device)

    labels_new = torch.zeros(N, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    if sources != None:
        sources_new = torch.zeros(N, device=device)
        i = 0
        for source in sources:
            sources_new[i] = source
            i += 1
    else:
        sources_new = torch.zeros(N, device=device)

    return molecule_words_new, molecule_atoms_new, molecule_adjs_new, proteins_new, protein_LM, molecule_LM, labels_new, sources_new

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}

        for name, param in model.named_parameters():
            self.shadow[name] = param.clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param

    def apply_shadow(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(self.shadow[name])

    def restore(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(self.shadow[name])

class Trainer(object):
    def __init__(self, model, batch_size, lr, weight_decay, ema):
        self.model = model
        self.ema = ema
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150, eta_min=0)
        self.batch_size = batch_size
        # self.optimizer = Ranger(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset, p_LMs, d_LMs, epoch):
        np.random.shuffle(dataset)
        N = len(dataset)

        loss_total = 0
        i = 0
        # self.optimizer = torch.nn.DataParallel(self.optimizer, device_ids=[0, 1])
        self.optimizer.zero_grad()

        molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, sources = [], [], [], [], [], [], [], []
        for data in dataset:
            i = i + 1
            molecule_word, molecule_atom, molecule_adj, protein, sequence, smile, label, source = data
            molecule_words.append(molecule_word)
            molecule_atoms.append(molecule_atom)
            molecule_adjs.append(molecule_adj)
            proteins.append(protein)
            sequences.append(sequence)
            smiles.append(smile)
            labels.append(label)
            sources.append(source)

            if i % self.batch_size == 0 or i == N:
                if len(molecule_words) != 1:
                    molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, sources = pack(molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, p_LMs, d_LMs, device, sources)
                    data = (molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, sources)
                    loss = self.model(data, epoch)#.mean()
                    # loss = loss / self.batch
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                    molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, sources = [], [], [], [], [], [], [], []
                else:
                    molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, sources = [], [], [], [], [], [], [], []
            else:
                continue

            if i % self.batch_size == 0 or i == N:
                self.optimizer.step()
                # self.schedule.step()
                self.optimizer.zero_grad()
                self.ema.update()
            loss_total += loss.item()
            # loss_total2 += loss3.item()

        return loss_total

class Tester(object):
    def __init__(self, model, batch_size, ema=None):
        self.model = model
        self.batch_size = batch_size
        self.ema = ema

    def test(self, dataset, p_LMs, d_LMs):
        if self.ema is not None:
            self.ema.apply_shadow()
        N = len(dataset)
        T, S, Y, S2, Y2, S3, Y3 = [], [], [], [], [], [], []
        i = 0
        molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels = [], [], [], [], [], [], []
        for data in dataset:
            i = i + 1
            molecule_word, molecule_atom, molecule_adj, protein, sequence, smile, label = data
            molecule_words.append(molecule_word)
            molecule_atoms.append(molecule_atom)
            molecule_adjs.append(molecule_adj)
            proteins.append(protein)
            sequences.append(sequence)
            smiles.append(smile)
            labels.append(label)

            if i % self.batch_size == 0 or i == N:
                # print(words[0])
                molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, _ = pack(molecule_words, molecule_atoms,
                                                                                       molecule_adjs, proteins, sequences, smiles, labels, p_LMs, d_LMs,
                                                                                       device)
                # print(words.shape)
                data = (molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, _)
                # print(self.model(data, train=False))
                correct_labels, ys1, ys2, ys3, _ = self.model(data, train=False)
                correct_labels = correct_labels.to('cpu').data.numpy()
                ys1 = ys1.to('cpu').data.numpy()
                ys2 = ys2.to('cpu').data.numpy()
                ys3 = ys3.to('cpu').data.numpy()
                predicted_labels1 = list(map(lambda x: np.argmax(x), ys1))
                predicted_scores1 = list(map(lambda x: x[1], ys1))
                predicted_labels2 = list(map(lambda x: np.argmax(x), ys2))
                predicted_scores2 = list(map(lambda x: x[1], ys2))
                predicted_labels3 = list(map(lambda x: np.argmax(x), ys3))
                predicted_scores3 = list(map(lambda x: x[1], ys3))

                for j in range(len(correct_labels)):
                    T.append(correct_labels[j])
                    Y.append(predicted_labels1[j])
                    S.append(predicted_scores1[j])
                    Y2.append(predicted_labels2[j])
                    S2.append(predicted_scores2[j])
                    Y3.append(predicted_labels3[j])
                    S3.append(predicted_scores3[j])

                molecule_words, molecule_atoms, molecule_adjs, proteins,  sequences, smiles, labels = [], [], [], [], [], [], []
            else:
                continue

        AUC = roc_auc_score(T, S)
        AUC2 = roc_auc_score(T, S2)
        AUC3 = roc_auc_score(T, S3)
        precision = precision_score(T, Y)
        # recall = recall_score(T, Y)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        return AUC, precision, PRC, AUC2, AUC3

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_select = "Da_to_Da"
    iteration = 120
    decay_interval = 5
    batch_size = 16
    lr = 5e-4
    weight_decay = 0.07
    lr_decay = 0.5
    layer_gnn = 3
    source_number = 3
    drop = 0.05
    setting = "Da_to_Da"
    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    dataset_train, dataset_test, p_LMs, d_LMs = data_load(data_select, device)
    train_set, val_set = train_test_split(dataset_train, test_size=0.2, random_state=42)
    setup_seed(2023)
    model = DGMM_DTI(layer_gnn=layer_gnn, source_number=source_number, device=device, dropout=drop).to(device)
    ema = EMA(model, decay=0.999)
    # model = torch.nn.DataParallel(model, device_ids=[1], output_device=1)
    # model = model.module.to(torch.device('cpu'))
    trainer = Trainer(model, batch_size, lr, weight_decay, ema)
    tester = Tester(model, batch_size, ema)

    """Output files."""
    file_AUCs = 'output/result/AUCs--' + setting + '.txt'
    file_model = 'output/model/' + setting
    AUCs = ('Epoch\tTime(sec)\tLoss_train\t'
            'AUC_test\tPrecision_test\tAUPR_test\tAUC_LM\tAUC_Sty')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()
    auc1 = 0
    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(train_set, p_LMs, d_LMs, epoch)
        AUC_val, precision_val, recall_val, AUC2, AUC3 = tester.test(val_set, p_LMs, d_LMs)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train,
                AUC_val, precision_val, recall_val, AUC2, AUC3]
        tester.save_AUCs(AUCs, file_AUCs)
        # tester.save_model(model, file_model)
        print('\t'.join(map(str, AUCs)))
        if auc1 < AUC_test:
            auc1 = AUC_test
            tester.save_model(file_model)
