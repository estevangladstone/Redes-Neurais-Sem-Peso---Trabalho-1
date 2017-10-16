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

tTraining_images = transform(training_images, 128)
tTesting_images = transform(testing_images, 128)

print("images loaded!")


def train_and_test(num_bits_addr, bleaching, b_bleaching, actv_threshold):
    start_time = time.time()
    w = WiSARD.WiSARD(num_bits_addr, bleaching, seed = int(time.time()), defaul_b_bleaching = b_bleaching)

    #print("beginning training...")

    # training discriminators
    w.fit(tTraining_images, training_labels)

    #print("training complete!")
    #print("beginning tests...")

    # predicting class
    #  Result will be a dictionary using the classes as key and the WiSARD result as values
    result = np.array(w.predict(tTesting_images))
    # Return how many items present in both lists are equal
    correct_items = np.sum(np.array(testing_labels) == result)

    #print("tests complete")
    #print("waiting results...")

    acc = float(correct_items) / float(len(testing_labels)) * 100
    time_elapsed = time.time() - start_time
    #res = np.array([actv_threshold, num_bits_addr, bleaching, b_bleaching, acc, time_elapsed])
    print(">>>> " + str(actv_threshold) +" | "+ str(num_bits_addr) +" | "+ str(bleaching) +" | "+ str(b_bleaching) +" || "+ str(acc) +" || "+ str(time_elapsed))

    #print(str(correct_items) + ' / ' + str(len(testing_labels)) + ' = ' + str( acc ) + '% de acerto')
    #print("Retina cell activation threshold: " + str(actv_threshold))
    #print("number of bits in address: " + str(num_bits_addr))
    #print("using bleaching: " + str(bleaching))
    #print("bleaching b: " + str(b_bleaching))
    #print("elapsed time: %s seconds" % (time.time() - start_time))

print(">>>> results format: actv. threshold | num_bits_addr | bleaching | b_bleaching || accuracy || time_elapsed")
#activation threshold from 64 to 192 (gray scale 0.25 - 0.75) increasing 32 by 32
for k in range(0, 5):
    actv = 64 + k*32
    tTraining_images = transform(training_images, actv)
    tTesting_images = transform(testing_images, actv)
#b_bleaching from 1 to 9 increasing 2 by 2 
    for j in range(0, 5):
        bb = 1 + j * 2
#num_bits_addr from 16 to 32 incrasing 2 by 2
        for i in range (0, 9):
            nba = 16 + i * 2
            train_and_test(nba, False, bb, actv)
            train_and_test(nba, True, bb, actv)
