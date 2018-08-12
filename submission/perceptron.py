from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron

import DataFormatting


def Format_inputs(vect, X):
    extra_features=DataFormatting.Tll(X)
    return ( DataFormatting.Vectorize(vect, X, extra_features) ).toarray()

def Benchmark(model, inputs, labels):
    y_pred = model.predict(inputs)
    y_actual = labels
    
    count=0
    loss_func=0
    for i in range(len(y_pred)):
        if (y_pred[i] != y_actual[i]):
            count += 1
            if y_actual==1 and y_pred[i]==-1:
                loss_func += 0.1
            elif y_actual[i]==-1 and y_pred[i]==1:
                loss_func += 1

    print(len(y_actual))
    print("count", count)

    test_sample_size = len(y_actual)
    E_out_of_sample = count*100/test_sample_size
    print(E_out_of_sample)
    print(loss_func)

def Run(X_train, X_test, y_train, y_test):
    vect = CountVectorizer(stop_words = 'english', min_df=0.0035)
    fit = vect.fit(X_train)

    X_train = Format_inputs(fit, X_train)
    X_test = Format_inputs(fit, X_test)

    model = Perceptron(max_iter=5, tol=None, penalty='l2', alpha=0.1).fit(X_train,y_train)

    Benchmark(model, X_train, y_train)
    print("-----------------------")
    Benchmark(model, X_test, y_test)