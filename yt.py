import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


cell_df = pd.read_csv('cell_samples.csv')
cell_df.tail()
cell_df.shape
cell_df.size
cell_df.count()
cell_df['Class'].value_counts()

maligmant_df = cell_df[cell_df['Class'] == 4][0:200]
benign_df = cell_df[cell_df['Class'] == 2][0:200]

axes = benign_df.plot(kind = 'scatter', x='Clump', y='UnifSize', color='blue', label='Benign')
maligmant_df.plot(kind = 'scatter', x='Clump', y='UnifSize', color='red', label='Maligmant', ax=axes)

# plt.show()


#identifying unwanted rows
cell_df.dtypes
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')

#REMOVE UNWANTED COLUMNS
cell_df.columns
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]

#cell_df 100 rows and 11 columns,
#picked 9 columns out of 11

#independent variable
x = np.asarray(feature_df)

#dependent variable
y = np.asarray(cell_df['Class'])
x[0:5]


#DIVIDE THE DATA AS TRAIN/TEST DATASET

# cell_df --> Train (80 rows) / Test (20 rows)
#Train (x,y) ... x itself is a 20 array ... y is 1D array
#Test (x,y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=4)

#546 x 9
x_train.shape
#546 x 1
y_train.shape
#137 x 9
x_test.shape
#137 x 1
y_test.shape

#MODELING (SVM with Scikit-learn)
from sklearn import svm
classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
classifier.fit(x_train, y_train)

#PREDICTION
y_predict = classifier.predict(x_test)


#EVALUATION
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))