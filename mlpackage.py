import pandas as pd
import scipy.stats as stats
import statistics
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
df = pd.read_csv("data.csv")
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
print(df.columns)
from sklearn.metrics import *
from stats import *
from sklearn.preprocessing import label_binarize
# x_axis= np.arrange(-100,100,0.01)
modDict = {}
df=df.drop("datasetId",axis=1)
df = df.replace("no stress",0)
df = df.replace("time pressure",1)
df = df.replace("interruption",2)
df = df.apply(pd.to_numeric)

tstDf = pd.read_csv("hrvtest.csv")
tstDf=tstDf.drop("datasetId",axis=1)
tstDf = tstDf.replace("no stress",0)
tstDf= tstDf.replace("time pressure",1)
tstDf= tstDf.replace("interruption",2)



tstDf = tstDf.apply(pd.to_numeric)
# def distance(x1, x2):
#     print(x1)
#     return np.sqrt(np.sum((x1 - x2) ** 2))
#     # selected_features = []
#     # for i in range(corr.shape[0]):
#     #   if corr.iloc[i,14] > threshold:
#     #     selected_features.append(df.iloc[:,i])
#     # return selected_features
def predictgaussianNB(dataset,row):
    summaries = summarize_by_class(dataset)
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    # for class_value, probability in probabilities.items():
    #     if best_label is None or probability > best_prob:
	# 		best_prob = probability
	# 		best_label = class_value
    # return best_label
    for class_value,probability in probabilities.items():
        if best_label is None or probability>best_prob:
            best_prob=probability
            best_label=class_value
    return best_label
THRESHOLD = 0.7
def checkcollinear(X,Y,THRESHOLD):
    corr = np.cov(X,Y)[0][1]/(np.std(X)*np.std(Y))
    if (corr >= THRESHOLD or corr <= -THRESHOLD):
        return True
    else:
        return False
plt.figure(figsize=(10,10))


# plt.show()
# def KNN(X_train,y_train,k,X):
#         y_pred=[]
#         print("X")
#         print(X)
#         for x in X:
#             print(x)
            
#             # distances = [distance(x, x_train) for x_train in X_train]
#             # k_idx = np.argsort(distances)[: k]
#             # # Extract the labels of the k nearest neighbor training samples
#             # labels = [y_train[i] for i in k_idx]
#             # # the most common class label
#             # most_common = Counter(labels).most_common(1)
#             # y_pred.append(most_common[0][0])
#         # return np.array(y_pred)


# def SVM(X_train, X_test, y_train, y_test ):
#     rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
#     poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
#     poly_pred = poly.predict(X_test)
#     rbf_pred = rbf.predict(X_test)
#     poly_accuracy = accuracy_score(y_test, poly_pred)
#     poly_f1 = f1_score(y_test, poly_pred, average='weighted')
#     print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
#     print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

#     poly_accuracy = accuracy_score(y_test, poly_pred)
#     poly_f1 = f1_score(y_test, poly_pred, average='weighted')
#     print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
#     print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

print("hello")
print(df)
features = list(df.head())
corr = df.corr()
import seaborn

s = seaborn.heatmap(corr,annot=False,cmap=plt.cm.Reds)
plt.show()
plt.savefig("heatmap.png")
# seaborn.displot(df,hist=False)
# plt.show()

print(features[:-1])
combination = combinations(features[:-1],2)
for i in list(combination):
    if(checkcollinear(df[i[0]], df[i[1]], THRESHOLD) and i[0] in features and i[1] in features):
        features.remove(i[1])
print(features)
df = df[features]
tstDf = tstDf[features]
print(df)
seperate(df)
print("---------")
for i in features:
    seaborn.distplot(df[i],bins=30)
    plt.show()
print("----------------")


# print(reduce(corr, 0.1))
"""


def KNN(X_train,y_train,k,X):
        y_pred=[]
        for x in X:
            distances = [distance(x, x_train) for x_train in X_train]
            k_idx = np.argsort(distances)[: k]
            # Extract the labels of the k nearest neighbor training samples
            labels = [y_train[i] for i in k_idx]
            # the most common class label
            most_common = Counter(labels).most_common(1)
            y_pred.append(most_common[0][0])
        return np.array(y_pred)

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)


k = 3
prediction=KNN(X_train,y_train,k,X_test)
"""
# df.plot(kind="scatter")
# plt.show()
for i in features[:-1]:
    seaborn.pairplot(df,hue="condition",size=2)
    
    for i in features[:-1]:
        seaborn.stripplot(x="condition", y=i, data=df, jitter=True, edgecolor="gray")
    
plt.show()
    

X = df[features[:-1]]
print(len(X))
Y = df["condition"].to_numpy()
print(Y)

l = LabelEncoder()
Y = l.fit_transform(Y)
X = (X-X.mean())/X.std()
X = X.to_numpy()
def distance (x,y):
    dist=0
    for i in range(len(x)):
        dist+=np.square(x[i]-y[i])
    return np.sqrt(dist)

    


def knn(X,Y,x,y,k):
    dist=[]
    for j in range(len(X)):
        dist.append((distance(X[j],x),Y[j]))
    dist.sort()
    k_dist = dist[:k]
    count=[0]*3
    for n in k_dist:
        count[n[1]]+=1
    return count.index(max(count))
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25)

print("pppppppppppppppppppppppp")

y_pred=[]
print(len(X_train))
print("-----------")
print(df)
k=int(np.sqrt(len(y_train)))
#print(k)
X_train=df[features[:-1]].to_numpy()
y_train=df[features[-1]].to_numpy()
X_test= tstDf[features[:-1]].to_numpy()
y_test=tstDf[features[-1]].to_numpy()
print(features)
for i in range(len(X_test)):
    print(i/500*100)
    print(knn(X_train,y_train,X_test[i],y_test[i],k))
    y_pred.append(knn(X_train,y_train,X_test[i],y_test[i],k))
    # y_pred.append(knn(df[features[:-1]].to_numpy(),df[features[-1]].to_numpy(),tstDf[features[:-1]].to_numpy(),tstDf[features[-1]].to_numpy(),k))
    
print(y_pred)
print("-----sssssssssssssssssssssssssssssssssssssssssssssssss")
print(type(df[features[-1]].to_numpy()))
ct = pd.crosstab(y_test, y_pred)
print(ct)
print(len(y_test))
accuracy = np.sum(y_test ==y_pred) / len(y_test)
        
print("KNN classification accuracy", accuracy)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=int(np.sqrt(len(y_train))))
Xx = df[features[:-1]]
print(len(X))
Yy = df["condition"].to_numpy()
print(Y)
model.fit(Xx, Yy)
p = model.predict(X_test)
from sklearn import metrics
modelAccuracy = metrics.accuracy_score(y_test, p)
print(modelAccuracy)
print("for K = ",int(np.sqrt(len(y_train))))
print('OUR MODEL ACCURACY : ',accuracy)
print('In built Accuracy : ',modelAccuracy)
proba = model.predict_proba(X_test)
percision = precision_score(y_test, p,average='micro')
print("PERCISION : - ",percision)

recall = recall_score(y_test, p,average='micro')
print("RECALL : ",recall)

print("FINALLY F-MEASURE",(2*percision*recall/(percision+recall)))



percision = precision_score(y_test, y_pred,average='micro')
print("PERCISION(OURS) : - ",percision)

recall = recall_score(y_test, y_pred,average='micro')
print("RECALL(OURS) : ",recall)

print("FINALLY F-MEASURE(OURS)",(2*percision*recall/(percision+recall)))
cmKNN = confusion_matrix(y_test,y_pred)
print(cmKNN)
val = []
val.append(percision)
val.append(recall)
val.append((2*percision*recall/(percision+recall)))

FP = cmKNN.sum(axis=0) - np.diag(cmKNN)  
FN = cmKNN.sum(axis=1) - np.diag(cmKNN)
TP = np.diag(cmKNN)
TN = cmKNN.sum() - (FP + FN + TP)

FPR = FP/(FP+TN)
#fpr,tpr,threshold = roc_curve(y_test,proba[:,1])
#rol_auc = metrics.auc(fpr,tpr)
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
import scikitplot as skplt

skplt.metrics.plot_roc_curve(y_test,proba)
plt.show()
auc = roc_auc_score(y_test,proba,multi_class='ovr')
model = summarize_by_class(df.values.tolist())
# tstDf = pd.read_csv("hrvtest.csv")
# tstDf=tstDf.drop("datasetId",axis=1)
# tstDf = tstDf.replace("no stress",0)
# tstDf= tstDf.replace("time pressure",1)
# tstDf= tstDf.replace("interruption",2)

print("AUC = ",auc)
# tstDf = tstDf[features]
# tstDf = tstDf.apply(pd.to_numeric)
val.append(auc)
modDict["KNN"]=val
Xval = tstDf[features[:-1]]
y_true = tstDf[features[-1]]
pred=[]
for i in range(len(Xval)):
    row=Xval.iloc[i]
    
    label = predict(model, row)
    label = int(label)
    if(label==0):
        print(label,y_true[i])
        print("Normal")
    elif(label==1):
        print(label,y_true[i])
        print("STRESS")
    else:
        print(label,y_true[i])
        print("INTERUPPTED")
    pred.append(label)

GNBaccr = metrics.accuracy_score(y_true, pred)
print("OURS========================")
print("ACCURACY",GNBaccr)
GNBpercision = precision_score(y_test, p,average='micro')
print("PERCISION : - ",GNBpercision)
GNBrecall = recall_score(y_test, p,average='micro')
print("RECALL : ",GNBrecall)
print("FINALLY F-MEASURE",(2*GNBpercision*GNBrecall/(GNBpercision+GNBrecall)))
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(df[features[:-1]], df[features[-1]])
GaussianNB()
print("SKLEARNS:-")
print(nb.score(Xval, y_true))
GNBpercision = precision_score(y_test, p,average='micro')

print("PERCISION : - ",GNBpercision)
GNBval=[]
GNBval.append(GNBpercision)
GNBrecall = recall_score(y_test, p,average='micro')
print("RECALL : ",GNBrecall)
GNBval.append(GNBrecall)
print("FINALLY F-MEASURE",(2*GNBpercision*GNBrecall/(GNBpercision+GNBrecall)))
GNBval.append((2*GNBpercision*GNBrecall/(GNBpercision+GNBrecall)))
GNBproba = nb.predict_proba(Xval)

skplt.metrics.plot_roc_curve(y_true,GNBproba)

GNBauc = roc_auc_score(y_true,GNBproba,multi_class='ovr')
print(GNBauc)
GNBval.append(GNBauc)
acount=0
bcount=0
for i in range(len(val)):
    if val[i]>GNBval[i]:
        acount= acount+1
    elif val[i]<GNBval[i]:
        bcount = bcount+1
if(max(acount,bcount)==acount):
    print("KNN IS BEST")
    print(val)

    
plt.show()



