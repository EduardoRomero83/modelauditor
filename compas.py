# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from torch.autograd import Variable
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

"""COMPAS.ipynb
Auditor of neural network to find disparate impact
"""

"""
!wget https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores.csv

"""


df = pd.read_csv("compas-scores.csv")
df = df.fillna(-1)
dropFeats = ["decile_score", "id", "name", "first",
             "last", "c_case_number", "r_case_number",
             "vr_case_number", "v_type_of_assessment",
             "type_of_assessment", "screening_date",
             "v_screening_date", "score_text"]
df = df.drop(dropFeats, axis=1)
df['compas_screening_date'] = df['compas_screening_date'].astype(
    'datetime64').astype(int)/100000000000
df['dob'] = df['dob'].astype('datetime64').astype(int)/100000000000
df['c_jail_in'] = df['c_jail_in'].astype('datetime64').astype(int)/100000000000
df['c_jail_out'] = df['c_jail_out'].astype(
    'datetime64').astype(int)/100000000000
df['c_offense_date'] = df['c_offense_date'].astype(
    'datetime64').astype(int)/100000000000
df['c_arrest_date'] = df['c_arrest_date'].astype(
    'datetime64').astype(int)/100000000000
df['r_offense_date'] = df['r_offense_date'].astype(
    'datetime64').astype(int)/100000000000
df['r_jail_in'] = df['r_jail_in'].astype(
    'datetime64').astype(int)/100000000000
df['r_jail_out'] = df['r_jail_out'].astype(
    'datetime64').astype(int)/100000000000
df['vr_offense_date'] = df['vr_offense_date'].astype(
    'datetime64').astype(int)/100000000000
df["sex"] = df["sex"].astype('category').cat.codes
df["age_cat"] = df["age_cat"].astype('category').cat.codes
df["race"] = df["race"].astype('category').cat.codes
df["c_charge_degree"] = df["c_charge_degree"].astype('category').cat.codes
df["c_charge_desc"] = df["c_charge_desc"].astype('category').cat.codes
df["r_charge_degree"] = df["r_charge_degree"].astype('category').cat.codes
df["r_charge_desc"] = df["r_charge_desc"].astype('category').cat.codes
df["vr_charge_degree"] = df["vr_charge_degree"].astype('category').cat.codes
df["vr_charge_desc"] = df["vr_charge_desc"].astype('category').cat.codes
df["v_score_text"] = df["v_score_text"].astype('category').cat.codes
pd.set_option("display.max_columns", None)

#y = df['decile_score.1']
#v_score_text has values (0, 1, 2, 3), but the number of examples corresponding to 0 is only 5 in the entire dataset.
#Nevertheless, the neural network is trained to be a 4-class classifier
y = df['v_score_text'].astype('int64')
ySex = df["sex"]
yRace = df["race"]
yAge = df["age"]
print("Sex distribution")
item_counts = df["sex"].value_counts()
print(item_counts)
print("Race distribution")
item_counts = df["race"].value_counts()
print(item_counts)
print("Age distribution")
item_counts = df["age"].value_counts()
print(item_counts)
protectedAttr = ["sex", "race", "dob", "age"]
X = df.drop(['decile_score.1', 'v_score_text', 'v_decile_score'], axis=1)
print(X.head())

cleanX = X.drop(protectedAttr, axis=1)
# returns a numpy array
x = cleanX.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
cleanX = pd.DataFrame(x_scaled)
cleanX.head()

# fixLablesDict = {
#     -1: 1,
#     }

# y = y.replace(fixLablesDict)
_, _, y_SexTrain, y_SexTest = train_test_split(
    cleanX, ySex, test_size=0.33, random_state=42)
_, _, y_RaceTrain, y_RaceTest = train_test_split(
    cleanX, yRace, test_size=0.33, random_state=42)
_, _, y_AgeTrain, y_AgeTest = train_test_split(
    cleanX, yAge, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    cleanX, y, test_size=0.33, random_state=42)


#Plot the histograms for the number of examples per class
"""plt.bar(df['v_score_text'].value_counts().index, df['v_score_text'].value_counts(), ec = 'black')
plt.title('Training Data')
plt.title('Full Data')
plt.xlabel('v_score_text (Target Variable)')
plt.ylabel('Number of examples')
"""

import matplotlib.pyplot as plt
plt.bar(y_train.value_counts().index, y_train.value_counts(), ec = 'black')
plt.title('Training Data')
plt.xlabel('v_score_text (Target Variable)')
plt.ylabel('Number of examples')


plt.bar(y_test.value_counts().index, y_test.value_counts(), ec = 'black')
plt.title('Training Data')
plt.title('Test Data')
plt.xlabel('v_score_text (Target Variable)')
plt.ylabel('Number of examples')



class normY(object):
    def __init__(self, bias=-1):
        self.bias = bias

    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        y = y + self.bias
        sample = x, y
        return sample


class Compas(Dataset):
    def __init__(self, trainX, trainY, transform=None):
        self.x = trainX.values
        self.y = trainY.values
        self.len = self.x.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = torch.Tensor(self.x[index].astype(float)), self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.len


ybias = normY()
compas = Compas(X_train, y_train, transform=None) 
trainloader = DataLoader(dataset=compas, batch_size=64,
                         shuffle=True)
#Referring to https://towardsdatascience.com/build-a-fashion-mnist-cnn-pytorch-style-efb297e22582 for the neural network, usage of cross entropy and the accuracy function 
def get_accuracy(model,dataloader, pr):
  count=0
  correct=0

  model.eval()
  with torch.no_grad():
    for batch in dataloader:
      people = batch[0]
      labels = batch[1]
      preds=model(people)
      #batch_correct=preds.argmax(dim=1).eq(labels-1).sum().item() + preds.argmax(dim=1).eq(labels).sum().item() + preds.argmax(dim=1).eq(labels+1).sum().item()
      batch_correct=preds.argmax(dim=1).eq(labels).sum().item()
#       if pr == True:
#         print(preds.argmax(dim=1))
      batch_count=len(batch[0])
      count+=batch_count
      correct+=batch_correct
  model.train()
  return correct/count

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
#         self.fc1 = nn.Linear(29, 25)
#         self.fc2 = nn.Linear(25, 20)
#         self.fc3 = nn.Linear(20, 15)
#         self.fc4 = nn.Linear(15, 10)

#         self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(27, 87)
        self.fc3 = nn.Linear(87, 40)
        self.fc4 = nn.Linear(40, 4)
        
        self.bn2 = nn.BatchNorm1d(num_features = 87)
        self.bn3 = nn.BatchNorm1d(num_features = 40)
        self.bn4 = nn.BatchNorm1d(num_features = 4)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.bn2(self.fc1(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = (F.relu(self.bn4(self.fc4(x))))
        # don't need softmax here since we'll use cross-entropy as activation.
        return x

model = Classifier()
optimizer = optim.Adam(model.parameters(), lr=0.009) 
model.train()
loader = trainloader
acc = list()
loss_ = list()
ep = list()
for epoch in range(20):
    
    for i, (people, labels) in enumerate(loader):
        people = Variable(people)
        labels = Variable(labels)
        #print(labels.unique())
        optimizer.zero_grad()
        outputs = model(people)

        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

    #Storing the accuracies and losses for plotting later
    a = get_accuracy(model,loader, False)
    acc.append(a)
    loss_.append(loss.item())
    ep.append(epoch)
    print('Epoch {0}: train set accuracy {1}'.format(epoch,a))

compas_test = Compas(X_test, y_test, None)
test_loader = DataLoader(dataset=compas_test, batch_size=128, shuffle = True)
                         #sampler=weighted_sampler)


t_loader = test_loader
print('test set accuracy {0}'.format(get_accuracy(model,t_loader, True)))

#Calculate, print and plot the confusion matrix
nb_classes = 4

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(t_loader):
        inputs = Variable(inputs)
        classes = Variable(classes)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

c_m = np.array(confusion_matrix)
#print(c_m)

#plt.imshow(c_m, cmap = 'cividis')
#plt.colorbar()



# model = Classifier()
# criterion = nn.NLLLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.003)

# for epoch in range(20):
#     for i, (people, labels) in enumerate(trainloader):
#         people = Variable(people)
#         labels = Variable(labels)

#         optimizer.zero_grad()
#         outputs = model(people)

#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         if (i+1) % 1000 == 0:
#             print('Epoch [%d/%d], Iter [%d] Loss: %.4f' % (epoch + 1, 10,
#                                                            i+1, loss.item()))

model.eval()
# # compasTest = Compas(cleanXTest, y_test, transform=ybias)
test = Variable(torch.Tensor(X_test.values))

pred = model(test)

_, predlabel = torch.max(pred.data, 1)

predlabel = predlabel.tolist()
predlabel = pd.DataFrame(predlabel)
predlabel.index = np.arange(3880) + 1
id = np.arange(3880) + 1
id = pd.DataFrame(id)
id.index = id.index + 1

predlabel = pd.concat([id, predlabel], axis=1)
predlabel.columns = ["Person", "Label"]
# predlabel.head()

model.eval()
# compasTest = Compas(cleanXTest, y_test, transform=ybias)
treeTrainSet = Variable(torch.Tensor(X_train.values))

yhat = model(treeTrainSet)

_, yhatLabel = torch.max(yhat.data, 1)

yhatlabel = yhatLabel.tolist()
yhatLabel = pd.DataFrame(yhatLabel)
yhatLabel.index = np.arange(7877) + 1
id2 = np.arange(7877) + 1
id2 = pd.DataFrame(id2)
id2.index = id2.index + 1

yhatLabel = pd.concat([id2, yhatLabel], axis=1)
yhatLabel.columns = ["Person", "Label"]
#print(yhatLabel.head())

del model

# Ground Truth biased
clf = tree.DecisionTreeClassifier()
X_train_bias, X_test_bias, y_train, y_test = train_test_split(X, y,
                                                              test_size=0.33,
                                                              random_state=42)
clf.fit(X_train_bias, y_train)

# tree.plot_tree(clf)
totalCorrect = 0
for i in range(len(X_test_bias)):
    predict = clf.predict([X_test_bias.iloc[i]])
    if predict[0] == y_test.iloc[i]:
        totalCorrect = totalCorrect + 1
print("Acc of biased decision tree learning ground truth: ", totalCorrect/3880)

del clf
del predict

# Ground Truth obscured
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# tree.plot_tree(clf)
totalCorrect = 0
for i in range(len(X_test)):
    predict = clf.predict([X_test.iloc[i]])
    if predict[0] == y_test.iloc[i]:
        totalCorrect = totalCorrect + 1
print("Acc of obscured decision tree learning ground truth: ", totalCorrect/3880)

del clf
del predict

# Predict sex
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_SexTrain)

# tree.plot_tree(clf)
totalCorrect = 0
for i in range(len(X_test)):
    predict = clf.predict([X_test.iloc[i]])
    if predict[0] == y_SexTest.iloc[i]:
        totalCorrect = totalCorrect + 1
print("Acc of decision tree learning protected attribute sex: ", totalCorrect/3880)

del clf
del predict

# Predict race
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_RaceTrain)

# tree.plot_tree(clf)
totalCorrect = 0
for i in range(len(X_test)):
    predict = clf.predict([X_test.iloc[i]])
    if predict[0] == y_RaceTest.iloc[i]:
        totalCorrect = totalCorrect + 1
print("Acc of decision tree learning protected attribute race: ", totalCorrect/3880)

del clf
del predict

# Predict age
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_AgeTrain)

# tree.plot_tree(clf)
totalCorrect = 0
for i in range(len(X_test)):
    predict = clf.predict([X_test.iloc[i]])
    if predict[0] == y_AgeTest.iloc[i]:
        totalCorrect = totalCorrect + 1
print("Acc of decision tree learning protected attribute age: ", totalCorrect/3880)

del clf
del predict

# Bias NN

auditor = tree.DecisionTreeClassifier(max_depth=15)
auditor.fit(X_train_bias, yhatLabel)
# tree.plot_tree(auditor)

totalCorrect = 0
for i in range(len(X_test_bias)):
    predict = auditor.predict([X_test_bias.iloc[i]])
    if predict[0][1] == predlabel.iloc[i]['Label']:
        totalCorrect = totalCorrect + 1
print("Acc of dec tree with bias feats learning nn: ", totalCorrect/3880)

name = "Auditor.dot"
with open(name, "w") as f:
    f = tree.export_graphviz(auditor, out_file=f)
    i = i + 1

del auditor
del predict

# Clean NN

auditorObscured = tree.DecisionTreeClassifier(max_depth=15)
auditorObscured.fit(X_train, yhatLabel)
# tree.plot_tree(auditorObscured)

totalCorrect = 0
for i in range(len(X_test)):
    predict = auditorObscured.predict([X_test.iloc[i]])
    if predict[0][1] == predlabel.iloc[i]['Label']:
        totalCorrect = totalCorrect + 1
print("Acc of obscured dec tree learning nn: ", totalCorrect/3880)

name = "AuditorObscured.dot"
with open(name, "w") as f:
    f = tree.export_graphviz(auditorObscured, out_file=f)
    i = i + 1

del auditorObscured
del predict

# Forest bias NN

forest = RandomForestClassifier(n_estimators=5, random_state=0,
                                max_depth=5, n_jobs=1)
forest.fit(X_train_bias, yhatLabel)

totalCorrect = 0
for i in range(len(X_test_bias)):
    predict = forest.predict([X_test_bias.iloc[i]])
    if predict[0][1] == predlabel.iloc[i]['Label']:
        totalCorrect = totalCorrect + 1
print("Acc Random Forest with bias learning nn: ", totalCorrect/3880)

i = 0
for estimator in forest.estimators_:
    name = "treeOut" + str(i) + ".dot"
    with open(name, "w") as f:
        f = tree.export_graphviz(estimator, out_file=f)
        i = i + 1

del forest
del predict

# Forest obscure NN

forestObscured = RandomForestClassifier(
    n_estimators=5, random_state=0, max_depth=5)
forestObscured.fit(X_train, yhatLabel)

totalCorrect = 0
for i in range(len(X_test)):
    predict = forestObscured.predict([X_test.iloc[i]])
    if predict[0][1] == predlabel.iloc[i]['Label']:
        totalCorrect = totalCorrect + 1
print("Acc Random Forest obscured learning nn: ", totalCorrect/3880)

i = 0
for estimator in forestObscured.estimators_:
    name = "treeOutObscured" + str(i) + ".dot"
    with open(name, "w") as f:
        f = tree.export_graphviz(estimator, out_file=f)
        i = i + 1

del forestObscured

