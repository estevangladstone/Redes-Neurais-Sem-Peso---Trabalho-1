from mnist import mnist
from PyWANN import WiSARD

mndata = MNIST('.')
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()

retina_length = 784
num_bits_addr = 14
bleaching = False

w = WiSARD.WiSARD(retina_length, num_bits_addr, bleaching)

# training discriminators
w.fit(training_images, training_labels)

# predicting class
result = w.predict(testing_images)  #  Result will be a dictionary using the classes as key and the WiSARD result as values

correct_items = set(testing_labels.items()) & set(result.items()) # Return only the items present in both lists

print (len(shared_items) + ' / ' + len(testing_labels) + ' = ' + len(shared_items)/len(testing_labels) + '\% de acerto')
