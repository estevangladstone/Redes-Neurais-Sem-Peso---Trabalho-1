from mnist import MNIST
from PyWANN import WiSARD
import time
import numpy as np

print("starting...")
print("loading images...")

def transform(retina, actv_threshold):
    ret = np.array(retina)
    for i in range(0, len(retina)):
        for j in range(0, len(retina[i])):
            if retina[i][j] > actv_threshold:
                ret[i][j] = 1
            else:
                ret[i][j] = 0
    return ret

mndata = MNIST('.')
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()

actv_threshold = 128
tTraining_images = transform(training_images, actv_threshold)
tTesting_images = transform(testing_images, actv_threshold)

print("images loaded!")

bleaching = True
b_bleaching = 1

start_time = time.time()
w = WiSARD.WiSARD(num_bits_addr, bleaching, seed = int(time.time()), defaul_b_bleaching = b_bleaching)

print("beginning training...")

# training discriminators
w.fit(tTraining_images, training_labels)

print("training complete!")
print("beginning tests...")

# Predicting class
# Result will be a dictionary using the classes as key and the WiSARD result as values
result = np.array(w.predict(tTesting_images))
# Return how many items present in both lists are equal
correct_items = np.sum(np.array(testing_labels) == result)

print("tests complete")
print("waiting results...")

acc = float(correct_items) / float(len(testing_labels)) * 100
time_elapsed = time.time() - start_time
   
print(str(correct_items) + ' / ' + str(len(testing_labels)) + ' = ' + str( acc ) + '% de acerto')
print("Retina cell activation threshold: " + str(actv_threshold))
print("number of bits in address: " + str(num_bits_addr))
print("using bleaching: " + str(bleaching))
print("bleaching b: " + str(b_bleaching))
print("elapsed time: %s seconds" % (time.time() - start_time))
