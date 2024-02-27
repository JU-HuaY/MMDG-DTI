# MMDG-DTI

Dependency
The code has been tested in the following environment:
Python-3.8,
PyTorch-1.8.0,
CUDA-11.1,

Dataset link: https://drive.google.com/file/d/1TLcGFcpmgFIPqm1aBw-EW23TdOy0z7rL/view?usp=share_link
BindingDB_Cold dataset: https://drive.google.com/file/d/1hSsu2D8XcRama_gnmp5FpxSEXi_wtbOV/view?usp=drive_link
Trained models link: https://drive.google.com/file/d/1dSpGwBbFp6yOPtClY4a-LbjvWJrowhdY/view?usp=sharing
Store the train model in a path "PATH_MODEL", the model could be loaded by the code "model_data = torch.load(PATH_MODEL)" in main.py

Data_process: The original data is first processed by the code document "Predata.py";
              Then we can use "data_split_name.py" to split the data sets; 
              After that, we can run main.py. This file will also import "merge.py" and "model.py";
              The only thing we have to do is to write the data protocols and the save path;
              
```python
>>> from transformers import AutoTokenizer, TFAutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("bert-base-uncased")
>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```
              The specific data protocols are described in the file "data_merge.py";

