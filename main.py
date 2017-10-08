from mnist import MNIST
from PyWANN import WiSARD
import time

print("starting...")
start_time = time.time()
print("loading images...")

mndata = MNIST('.')
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()

print("images loaded!")

retina_length = 28*28
num_bits_addr = 10
bleaching = True

w = WiSARD.WiSARD(num_bits_addr, bleaching)

print("beginning training...")

# training discriminators
w.fit(training_images, training_labels)

print("training complete!")
print("beginning tests...")

# predicting class
result = w.predict(testing_images)  #  Result will be a dictionary using the classes as key and the WiSARD result as values

correct_items = set(testing_labels) & set(result) # Return only the items present in both lists

print("tests complete")
print("waiting results...")

acc = float(len(correct_items)) / float(len(testing_labels)) * 100
print(str(len(correct_items)) + ' / ' + str(len(testing_labels)) + ' = ' + str( acc ) + '% de acerto')
print("number of bits in address: " + str(num_bits_addr))
print("using bleaching: " + str(bleaching))
print("elapsed time: %s seconds" % (time.time() - start_time))
