#utf-8
from NMF_model import NMF_model
import pandas as pd
import profile
import time

data_df = pd.read_csv("./Data/train_data.txt", sep='\t', header=-1)
test_df = pd.read_csv("./Data/test_data.txt", sep='\t', header=-1)

model = NMF(data_df, dimk=50, idf_w=1, Lambda=5)
model.decomposition()
model.evaluation(test_df, topK=6183)
