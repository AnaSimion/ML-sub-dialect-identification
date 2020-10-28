import numpy as np
from sklearn import preprocessing
import codecs
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

#modelul bag-of-words laboratorul 5
class Bag_of_words:

    def __init__(self):
        self.vocabulary = {}
        self.words = []
        self.vocabulary_length = 0

    def build_vocabulary(self, data):
        for document in data:
            for word in document:
                # word = word.lower()
                if word not in self.vocabulary.keys():
                    self.vocabulary[word] = len(self.vocabulary)
                    self.words.append(word)

        self.vocabulary_length = len(self.vocabulary)
        self.words = np.array(self.words)

    def get_features(self, data):
        features = np.zeros((len(data), self.vocabulary_length))

        for document_idx, document in enumerate(data):
            for word in document:
                if word in self.vocabulary.keys():
                    features[document_idx, self.vocabulary[word]] += 1
        return features


#citirea datelor

f = open('train_samples.txt', 'r')
train_samples = [list(line.split()) for line in f]

#eliminam id-ul din datele de antrenare
for i in train_samples:
    i.pop(0)

f = open('test_samples.txt', 'r')
test_samples = [list(line.split()) for line in f]

for i in test_samples:
    i.pop(0)

train_labels = np.loadtxt('train_labels.txt', 'int')

a = np.zeros(len(train_labels))

#pastram doar 0/1 din etichete
for elemnet in range(len(train_labels)):
    a[elemnet] = train_labels[elemnet][1]

f = open('validation_samples.txt', 'r')
validation_samples = [list(line.split()) for line in f]

validation_labels = np.loadtxt('validation_labels.txt', 'int')
b = np.zeros(len(validation_labels))

for elemnet in range(len(validation_labels)):
    b[elemnet] = validation_labels[elemnet][1]


np_load_old = np.load

np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

np.load = np_load_old

bow_model = Bag_of_words()
bow_model.build_vocabulary(train_samples)

#normalizarea datelor din laboratorul 5
def normalize_data(train_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')

    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        return (scaled_train_data, scaled_test_data)
    else:
        print("No scaling was performed. Raw data is returned.")
        return (train_data, test_data)

#normalizarea datelor
train_features = bow_model.get_features(train_samples)
test_features = bow_model.get_features(test_samples)
scaled_train_data, scaled_test_data = normalize_data(train_features, test_features, type='l2')

from sklearn import svm
#Definirea modelului Antrenarea modelului Prezicerea etichetelor

#aici am avut diverse incercari precum:
#(C=2, kernel='linear', gamma='auto'), (C=2, decision_function_shape=�ovo�) 
svm_model = svm.SVC(C=1, kernel='linear')
#am avut si o incercare cu model = LinearDiscriminantAnalysis()
svm_model.fit(scaled_train_data, a)
predicted_labels_svm = svm_model.predict(scaled_test_data)

#citirea inca o data a test_samples pentru a extrage indexul pe care il afisez in fisierul cu solutia
fileTest = codecs.open('test_samples.txt', encoding= 'utf-8')
test_samples = np.genfromtxt(fileTest, delimiter='\t', dtype=None, names=('ID', 'Text'))

#afisarea sub forma ceruta
a_file = open("test.txt", "w")
a_file.write('id,label\n')
k = 0
for row in predicted_labels_svm:
    a_file.write(str(int(test_samples['ID'][k])))
    a_file.write(',')
    a_file.write(str(int(row)))
    a_file.write('\n')
    k = k + 1

a_file.close()

#obtinerea confusion_matrix si f1_score pentru datele de validare
train_features = bow_model.get_features(train_samples)
test_features = bow_model.get_features(validation_samples)
scaled_train_data, scaled_test_data = normalize_data(train_features, test_features, type='l2')
svm_model = svm.SVC(C=1, kernel='linear')
svm_model.fit(scaled_train_data, a)
predicted_labels_svm = svm_model.predict(scaled_test_data)

from sklearn.metrics import confusion_matrix
print("Confusion matrix: ")
print(confusion_matrix(b, predicted_labels_svm))

from sklearn.metrics import f1_score
print('\n' + "F1_score: ")
print(f1_score(b, predicted_labels_svm))
