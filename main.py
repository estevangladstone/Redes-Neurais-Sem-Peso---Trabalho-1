from mnist import MNIST
from PyWANN import WiSARD
import time
import numpy as np
from random import randint

print("starting...")
start_time = time.time()
print("loading images...")

activation = 127

def transform(retina):
    for i in range(0, len(retina)):
        for j in range(0, len(retina[i])):
            if retina[i][j] > activation:
                retina[i][j] = 1
            else:
                retina[i][j] = 0
    return retina

mndata = MNIST('.')
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()

training_images = transform(training_images)
testing_images = transform(testing_images)

print("images loaded!")

num_bits_addr = 28
bleaching = True

w = WiSARD.WiSARD(num_bits_addr, bleaching, seed = int(time.time()))

print("beginning training...")

# training discriminators
w.fit(training_images, training_labels)

print("training complete!")
print("beginning tests...")

# predicting class
result = np.array(w.predict(testing_images))  #  Result will be a dictionary using the classes as key and the WiSARD result as values
correct_items = np.sum(np.array(testing_labels) == result) # Return only the items present in both lists


print("tests complete")
print("waiting results...")

acc = float(correct_items) / float(len(testing_labels)) * 100
print(str(correct_items) + ' / ' + str(len(testing_labels)) + ' = ' + str( acc ) + '% de acerto')
print("Retina cell activation threshold: " + str(activation))
print("number of bits in address: " + str(num_bits_addr))
print("using bleaching: " + str(bleaching))
print("elapsed time: %s seconds" % (time.time() - start_time))
