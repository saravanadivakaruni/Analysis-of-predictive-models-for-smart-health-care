import pandas as pd
data = pd.read_csv("ecoli")
data.columns=[]
data.to_csv(".csv",index=None,header=True)
