from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

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
    vect = CountVectorizer(stop_words = 'english', min_df = 0.0035).fit(X_train)

    X_train = Format_inputs(vect, X_train)
    X_test = Format_inputs(vect, X_test)

    model = MLPClassifier()
    parameters = [{'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}]
    print(model.get_params().keys())
    
    model = GridSearchCV(model, parameters, cv = 10)
    model.fit(X_train, y_train)
    best_accuracy = model.best_score_
    best_parameters = model.best_params_
    print(best_accuracy)
    print(best_parameters)
    """
    Benchmark(model, X_train, y_train)
    print("-----------------------")
    Benchmark(model, X_test, y_test)
    """