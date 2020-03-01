import csv
import random
import MLP_Classification
import matplotlib.pyplot as plt

csvfile = "ionosphere_csv.csv"
data = []
data_suffle = []
target = []

lr = 0.9
n_train = 80
n_epoch = 500


with open(csvfile, 'r') as r, open('file_shuffled.csv', 'w') as w:
    data = r.readlines()
    header, rows = data[0], data[1:]
    random.shuffle(rows)
    rows = '\n'.join([row.strip() for row in rows])
    w.write(header +rows)

with open('file_shuffled.csv','r') as f:
    with open('file_data.csv','w') as fout:
        next(f)
        writer = csv.writer(fout)
        for row in csv.reader(f):
            writer.writerow(row[:-1])

with open('file_data.csv', 'r') as f:
    my_list = [[float(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
    data = [x for x in my_list if x != []]

with open('file_shuffled.csv','r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        target.append([(row['class'])])

target = [[1,0] if element == ['g'] else [0,1] if element == ['b'] else element for element in target]
 
    
print("Number of data: ",len(data))
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
    target_i = []
    for j in range(0,len(data[i])):
        train_i.append(data[i][j])
    train_data.append(train_i)
    train_target.append(target[i])

for i in range(train_index,len(data)):
    test_i = []
    for j in range(0,len(data[i])):
        test_i.append(data[i][j])
    test_data.append(test_i)
    test_target.append(target[i])

network = MLP_Classification.MLP(len(data[0]),2,10,lr)
print("\nStart Training... \n")
train_result = network.training(train_data,n_epoch,train_target)

train_result  = ['g' if element == [1,0] else 'b' if element == [0,1] else element for element in train_result]
train_target = ['g' if element == [1,0] else 'b' if element == [0,1] else element for element in train_target]
with open('train_prediction.csv', 'w') as csvtrain:
    train_result_csv = csv.writer(csvtrain, delimiter=',')
    train_result_csv.writerow(['Predicted', 'Target'])
    for i in range(0,len(train_result)):
        row = []
        row.append(train_result[i])
        row.append(train_target[i])
        train_result_csv.writerow(row)


plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Errors')
plt.title('Training curve')
plt.plot(network.lossFunction()) 
plt.savefig("Learning curve.png")

print("\nStart Testing... \n")
test_result = network.testing(test_data,test_target)

test_result  = ['g' if element == [1,0] else 'b' if element == [0,1] else element for element in test_result]
test_target = ['g' if element == [1,0] else 'b' if element == [0,1] else element for element in test_target]
with open('test_prediction.csv', 'w') as csvtest:
    test_result_csv = csv.writer(csvtest, delimiter=',')
    test_result_csv.writerow(['Predicted', 'Target', 'State'])
    for i in range(0,len(test_result)):
        row = []
        row.append(test_result[i])
        row.append(test_target[i])
        if test_result[i] == test_target[i]:
            row.append('True')
        else : row.append('False')
        test_result_csv.writerow(row)

