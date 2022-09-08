from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np 

def process_y(y_values_pd):
    return np.array(y_values_pd.values).astype(int)

def split_data_cellclass(adata, n_train, target_col='leiden'):
    if n_train < 1: 
        n_train = int(adata.shape[0] * n_train)
    x_train, x_test = adata.X[:n_train, :], adata.X[n_train:, :]
    y_train, y_test = adata.obs.iloc[:n_train][target_col], adata.obs.iloc[n_train:][target_col]
    return x_train, x_test, process_y(y_train), process_y(y_test) 

def svm_cellclass(x_train, y_train, x_test=None, y_test=None, kernel='linear'):
    clf = SVC(kernel=kernel)
    clf.fit(x_train,y_train)
    if x_test is not None and y_test is not None: 
        y_pred = clf.predict(x_test)
        print('SVM test accuracy:', 100*accuracy_score(y_test,y_pred))
    return clf
