#Libraries required
import pandas 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Loading dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class_type']
dataset = pandas.read_csv(url, names=names)

#checking inconsistency in data
dataset.info()

#summarising data

#shape of data
print(dataset.shape)

#displaying some data
print(dataset.head(20))

#statistical summary of dataset
print(dataset.describe())

#class description
print(dataset.groupby('class_type').size())

#visualising data

#scatter plot sepal length VS width 
fig = dataset[dataset.class_type=='Iris-setosa'].plot(kind='scatter',x='sepal-length',y='sepal-width',color='orange', label='Setosa')
dataset[dataset.class_type=='Iris-versicolor'].plot(kind='scatter',x='sepal-length',y='sepal-width',color='blue', label='versicolor',ax=fig)
dataset[dataset.class_type=='Iris-virginica'].plot(kind='scatter',x='sepal-length',y='sepal-width',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

#scatter plot petal length VS width 
fig = dataset[dataset.class_type=='Iris-setosa'].plot(kind='scatter',x='petal-length',y='petal-width',color='orange', label='Setosa')
dataset[dataset.class_type=='Iris-versicolor'].plot(kind='scatter',x='petal-length',y='petal-width',color='blue', label='versicolor',ax=fig)
dataset[dataset.class_type=='Iris-virginica'].plot(kind='scatter',x='petal-length',y='petal-width',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

# box and whisker plots
fig=dataset.plot(kind='box', subplots=True, layout=(2,2),sharex=False, sharey=False)
fig=plt.gcf()
fig.set_size_inches(8,8)
plt.show()

# histograms
dataset.hist(edgecolor='black',linewidth=4)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()

#heatmap
plt.figure(figsize=(8,8)) 
sns.heatmap(dataset.corr(),annot=True)
plt.show()

#validation dataset
X=dataset[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
Y=dataset.class_type
train_X, val_X, train_Y, val_Y = train_test_split(X, Y,random_state=1,test_size=0.3)

#checking different algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC()))

# making predictions
for name,algorithm in models:
    model= algorithm
    model.fit(train_X, train_Y)
    prediction = model.predict(val_X)
    print('The accuracy of the %s is %f:'%(name,accuracy_score(prediction,val_Y)))
print('\n')

#petals training data
petal_X=dataset[['petal-length', 'petal-width']]
petal_Y=dataset.class_type
train_XP, val_XP, train_YP, val_YP = train_test_split(petal_X, petal_Y,random_state=1,test_size=0.3)

#sepals training data
sepal_X=dataset[['sepal-length', 'sepal-width']]
sepal_Y=dataset.class_type
train_XS, val_XS, train_YS, val_YS = train_test_split(sepal_X, sepal_Y,random_state=1,test_size=0.3)

for name,algorithm in models:
    model=algorithm
    model.fit(train_XP,train_YP) 
    prediction=model.predict(val_XP) 
    print('The accuracy of the %s using Petals is: %f' %(name,accuracy_score(prediction,val_YP)))
    model.fit(train_XS,train_YS) 
    prediction=model.predict(val_XS) 
    print('The accuracy of the %s using Sepal is: %f' %(name,accuracy_score(prediction,val_YS)))
    print('\n')

