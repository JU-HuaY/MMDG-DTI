# MMDG-DTI

Dependency
The code has been tested in the following environment:
Python-3.8,
PyTorch-1.8.0,
CUDA-11.1,

Dataset link: https://drive.google.com/file/d/1TLcGFcpmgFIPqm1aBw-EW23TdOy0z7rL/view?usp=share_link
BindingDB_Cold dataset: https://drive.google.com/file/d/1hSsu2D8XcRama_gnmp5FpxSEXi_wtbOV/view?usp=drive_link
Trained models link: https://drive.google.com/file/d/1dSpGwBbFp6yOPtClY4a-LbjvWJrowhdY/view?usp=sharing
Store the train model in a path "PATH_MODEL", the model could be loaded by the following code in main.py
```python
>>> model_data = torch.load(PATH_MODEL)
```
Data_process: The original data is first processed by the code document "Predata.py";
              Then we can use "data_split_name.py" to split the data sets; 
              After that, we can run main.py. This file will also import "merge.py" and "model.py";
              The only thing we have to do is to write the data protocols and the save path;
              
```python
>>> data_select = "D_H_C_to_B"
>>> setting = "D_H_C_to_B"
>>> file_AUCs = 'output/result/AUCs--' + setting + '.txt'
>>> file_model = 'output/model/' + setting
```
              The specific data protocols are described in the file "data_merge.py";

