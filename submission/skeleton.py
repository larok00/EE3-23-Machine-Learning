def LoadData():
    filename = 'data/SMSSpamCollection.txt'
    file = open(filename, 'r')
    
    X = []
    y = []

    for entries in file:
        X.append(entries.split('	')[1])  # list of X
        y.append(entries.split('	')[0])  # list of y
    file.close()

    return X, y

def Binarize(labels):
    newLabels=[]
    for l in labels:
        newLabels.append(1) if l=='spam' else newLabels.append(-1)
    return newLabels

####################################################################################################
####################################################################################################

from sklearn.model_selection import train_test_split #for splitting dataset

X, y = LoadData()
y=Binarize(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=1)

import svm
svm.Run(X_train, X_test, y_train, y_test)

