import pandas as pd

ds = pd.read_csv('C:/Users/MFBA/Documents1/big data dp project/Dataset/steam_reviews.csv')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# preprocessing
le.fit(["Recommended", "Not Recommended"])
Y  = le.fit_transform(ds['recommendation'])
ds = ds.drop(['recommendation'],axis=1)
ds['recommendation'] = Y

le2 =  LabelEncoder()
le2.fit(['True', 'False'])
Y2 = le.fit_transform(ds['is_early_access_review'])
ds = ds.drop(['is_early_access_review'],axis=1)
ds['is_early_access_review'] = Y2

le3 =  LabelEncoder()
le3.fit(ds['title'])
Y3 = le.fit_transform(ds['title'])
ds = ds.drop(['title'],axis=1)
ds['title'] = Y3

#oversampling technique
from sklearn.utils import resample
df_majority = ds[ds.recommendation==1]
df_minority = ds[ds.recommendation==0]
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=303593,    # to match majority class
                                 random_state=123)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
print(df_upsampled['recommendation'].value_counts())


df_upsampled.to_csv(r'C:\Users\MFBA\Documents1\big data dp project\Dataset\out.csv', index = False)