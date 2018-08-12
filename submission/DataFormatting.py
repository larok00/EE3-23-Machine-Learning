from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

import re
from scipy.sparse import coo_matrix,hstack

def Tll(X):
    """ tokenize

        lowercase

        lemmatize   """

    extra_features=[]

    for i in range(len(X)):
        caps = re.compile(r'[A-Z][A-Z]+')
        exclam = re.compile(r'!!+')
        question = re.compile(r'\?\?+')
        money = re.compile(r'\$|£|€')
        hashtag = re.compile(r'#')
        percent = re.compile(r'%')

        consecutive_list = [caps,exclam, question]
        single_list =[money, hashtag, percent]
        temp=[]
        for pattern in consecutive_list:
            total=0
            lst=pattern.findall(X[i])
            for x in lst:
                total += len(x)
            total_ratio=float(100*total/len(X[i]))
            if len(lst)!=0:
                average_ratio=float(100*total/(len(lst)*len(X[i])))
                max_ratio=float(100*len(max(lst, key=len))/len(X[i]))
            else:
                average_ratio=0.0
                max_ratio=0.0

            temp.append(total_ratio)
            temp.append(average_ratio)
            temp.append(max_ratio)

        for pattern in single_list:
            total=0
            lst=pattern.findall(X[i])
            for x in lst:
                total += len(x)
            total_ratio=float(100*total/len(X[i]))

            temp.append(total_ratio)

        extra_features.append(temp)

        x_lowercase = X[i].lower()

        #getting rid of both spaces and punctuation marks
        X[i] = RegexpTokenizer(r'\w+').tokenize(x_lowercase)


        for x in X[i]:
            x = WordNetLemmatizer().lemmatize(x, pos="v")
    
    return coo_matrix(extra_features)


def Vectorize(fit, X, coo_mat):
    #list of tokens into strings
    for i in range(len(X)):
        X[i] = ' '.join(X[i])
    print(X[1])
    #A feature vector matrix whose rows denote number of samples of training set and columns denote words of dictionary. 
    #The value at index [i][j] will be the number of occurrences of jth word of dictionary in ith file.
    transformed = fit.transform(X)
    return ( hstack([transformed, coo_mat]) )