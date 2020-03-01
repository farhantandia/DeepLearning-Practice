import csv
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import MLP_Regression

lr = 0.0000002
n_train = 75
n_epoch = 7000 

print('Number of Epoch :',n_epoch) 

csvfile = "EnergyEfficiency_data.csv"
data_df = pd.read_csv(csvfile)

data_df.head()
features = data_df.columns[:-2]

traindf = pd.DataFrame(data_df,columns=features)

feature_name = list(traindf.columns)


X = traindf.copy()
y = data_df['Heating Load']

#number of features that use for the model (the top n)
num_feats=4
X.columns.tolist()

def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    
    # cal the correlation between each features and label data
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
        
    # change NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    
    # get the feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
   
    # the state of feature selection result, 0 for not used, 1 for used
    cor_support = [True if i in cor_feature else False for i in feature_name]
    
    return cor_support, cor_feature
cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')
print('Input Features :',cor_feature)

target =y.to_string(header=False,index=False).split('\n')
target = [[float(y) for y in x.split(',')] for x in target]
# print(target)

train = pd.DataFrame(data_df,columns=cor_feature)
x = train.to_string(header=False,index=False,index_names=False).split('\n')
vals = [','.join(ele.split()) for ele in x]
data = [[float(y) for y in x.split(',')] for x in vals]
# print(len(data))

for i in range(0,len(data)):
    for j in range(0,len(target[i])):
        data[i].append(target[i][j])

data_shuffled = data
'''Split data to training data and testing data'''
train_index = int((n_train * len(data))/100)
print("Number of data training: ",train_index)
test_index = len(data) - train_index
print("Number of data testing:",test_index)

train_data = []
test_data = []
train_target = []
test_target = []

for i in range(0,train_index):
    train_i = []
    for j in range(0,len(data[i])-1):
        train_i.append(data[i][j])
    train_data.append(train_i)
    train_target.append(data[i][-1])

for i in range(train_index,len(data)):
    test_i = []
    for j in range(0,len(data[i])-1):
        test_i.append(data[i][j])
    test_data.append(test_i)
    test_target.append(data[i][-1])

'''Shuffle data for training'''
random.shuffle(data_shuffled)
train_index2 = int((n_train * len(data_shuffled))/100)
print("Number of data shuffled for training: ",train_index2)
test_index2 = len(data) - train_index
print("Number of data shuffled for testing:",test_index2)

train_data_shuffled = []
test_data_shuffled = []
train_target_shuffled = []
test_target_shuffled = []

for i in range(0,train_index2):
    train_i = []
    for j in range(0,len(data_shuffled[i])-1):
        train_i.append(data_shuffled[i][j])
    train_data_shuffled.append(train_i)
    train_target_shuffled.append(data_shuffled[i][-1])

for i in range(train_index2,len(data_shuffled)):
    test_i = []
    for j in range(0,len(data_shuffled[i])-1):
        test_i.append(data_shuffled[i][j])
    test_data_shuffled.append(test_i)
    test_target_shuffled.append(data_shuffled[i][-1])

'''Put the shuffled training data to MLP for training'''

network = MLP_Regression.MLP(len(data_shuffled[0])-1,2,10,lr)
print("\nStart Training... \n")
train_shuffled_result = network.training(train_data_shuffled,n_epoch,train_target_shuffled)

'''Plot the Learning Curve and save the prediciton'''
plt.xlabel('Epoch')
plt.ylabel('Root Mean Square Errors')
plt.title('Training curve')
plt.plot(network.lossFunction())
plt.savefig("Learning curve.png")

train_shuffled_result_temp = []
train_shuffled_target_temp = []
for i in range(0,len(train_shuffled_result)):
    train_shuffled_result_temp.append(train_shuffled_result[i])
    train_shuffled_target_temp.append(train_target_shuffled[i])

with open('train_prediction.csv', 'w') as csvtrain:
    train_result_csv = csv.writer(csvtrain, delimiter=',')
    train_result_csv.writerow(['Predicted', 'Target'])
    for i in range(0,len(train_shuffled_result)):
        row = []
        row.append(train_shuffled_result[i])
        row.append(train_target_shuffled[i])
        train_result_csv.writerow(row)


'''Put the normal train and test data to MLP for testing'''
print("\nStart Testing...")
print (" Testing order :\n1. shuffled testing data, \n2. unshuffled testing data, \n3. unshuffled training data ")
test_shuffled_result = network.testing(test_data_shuffled,test_target_shuffled)
test_result = network.testing(test_data,test_target)
test_train_result = network.testing(train_data,train_target)

test_shuffled_result_temp = []
test_shuffled_target_temp = []
for i in range(0,len(test_result)):
    test_shuffled_result_temp .append(test_shuffled_result[i])
    test_shuffled_target_temp.append(test_target_shuffled[i])

test_result_temp = []
test_target_temp = []
for i in range(0,len(test_result)):
    test_result_temp.append(test_result[i])
    test_target_temp.append(test_target[i])

test_train_result_temp = []
test_train_target_temp = []
for i in range(0,len(test_train_result)):
    test_train_result_temp.append(test_train_result[i])
    test_train_target_temp.append(train_target[i])

with open('test_prediction.csv', 'w') as csvtest:
    test_shuffled_result_csv = csv.writer(csvtest, delimiter=',')
    test_shuffled_result_csv.writerow(['Predicted', 'Target'])
    for i in range(0,len(test_result)):
        row = []
        row.append(test_shuffled_result[i])
        row.append(test_target_shuffled[i])
        test_shuffled_result_csv.writerow(row)

with open('ori_test_prediction.csv', 'w') as csvtest:
    test_result_csv = csv.writer(csvtest, delimiter=',')
    test_result_csv.writerow(['Predicted', 'Target'])
    for i in range(0,len(test_result)):
        row = []
        row.append(test_result[i])
        row.append(test_target[i])
        test_result_csv.writerow(row)

with open('ori_train_prediction.csv', 'w') as csvtest:
    test_train_result_csv = csv.writer(csvtest, delimiter=',')
    test_train_result_csv.writerow(['Predicted', 'Target'])
    for i in range(0,len(test_train_result)):
        row = []
        row.append(test_train_result[i])
        row.append(train_target[i])
        test_train_result_csv.writerow(row)


'''Plot the training prediction and labels'''
plt.figure(figsize=(15, 7))
plt.title("Prediction for shuffled training data")
plt.xlabel('#th case')
plt.ylabel('heating load')
plt.plot(train_shuffled_result_temp, color="red", label="Predicted")
plt.plot(train_shuffled_target_temp, color="blue", label="Label")
plt.legend(loc="upper left")
plt.savefig("training.png")
plt.grid(True)
plt.show()

'''Plot the testing prediction and labels'''
plt.figure(figsize=(15, 7))
plt.title("Prediction for shuffled testing data")
plt.xlabel('#th case')
plt.ylabel('heating load')
plt.plot(test_shuffled_result_temp, color="red", label="Predicted")
plt.plot(test_shuffled_target_temp, color="blue", label="Label")
plt.legend(loc="upper left")
plt.savefig("testing_test_data.png")
plt.grid(True)

plt.figure(figsize=(15, 7))
plt.title("Prediction for unshuffled testing data")
plt.xlabel('#th case')
plt.ylabel('heating load')
plt.plot(test_result_temp, color="red", label="Predicted")
plt.plot(test_target_temp, color="blue", label="Label")
plt.legend(loc="upper left")
plt.savefig("testing_test_data.png")
plt.grid(True)

plt.figure(figsize=(15, 7))
plt.title("Prediction for unshuffled training data")
plt.xlabel('#th case')
plt.ylabel('heating load')
plt.plot(test_train_result_temp, color="red", label="Predicted")
plt.plot(test_train_target_temp, color="blue", label="Label")
plt.legend(loc="upper left")
plt.savefig("testing_train_data.png")
plt.grid(True)

'''To get more readable results of model prediction using exp smoothing'''
def exponential_smoothing(series,series2, alpha):
   
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result
    result2 = [series2[0]] # first value is same as series
    for n in range(1, len(series)):
        result2.append(alpha * series2[n] + (1 - alpha) * result2[n-1])
    return result2

def plotExponentialSmoothing_training(series,series2, alpha):
   
    plt.figure(figsize=(15, 7))
    plt.plot(exponential_smoothing(series,series2, alpha),"r", label="Prediction with Exp Smoothing(Alpha {})".format(alpha))
    plt.plot(exponential_smoothing(series2,series, alpha),"b", label="Label with Exp Smoothing(Alpha{})".format(alpha))
#         plt.plot(series, "g", label = "Actual prediction")
#         plt.plot(series2, "y", label = "Actual label")
    plt.legend(loc="best")
    plt.xlabel('#th case')
    plt.ylabel('heating load')
    plt.axis('tight')
    plt.title("Exponential Smoothing for shuffled training data prediction")
    plt.grid(True)
    plt.savefig('Exp training.png')
    plt.show()
plotExponentialSmoothing_training(train_shuffled_result_temp,train_shuffled_target_temp,0.1)

def plotExponentialSmoothing_testing(series,series2, alpha):
   
    plt.figure(figsize=(15, 7))
    plt.plot(exponential_smoothing(series,series2, alpha),"r", label="Prediction with Exp Smoothing(Alpha {})".format(alpha))
    plt.plot(exponential_smoothing(series2,series, alpha),"b", label="Label with Exp Smoothing(Alpha{})".format(alpha))

    plt.legend(loc="best")
    plt.xlabel('#th case')
    plt.ylabel('heating load')
    plt.axis('tight')
    plt.title("Exponential Smoothing for shuffled testing data prediction")
    plt.grid(True)
    plt.savefig('Exp testing.png')
    plt.show()
plotExponentialSmoothing_testing(test_shuffled_result_temp,test_shuffled_target_temp,0.1)
