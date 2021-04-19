from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator; '''for generating batches of tensor image with real-time data augmentation '''
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.regularizers import l2

from keras.initializers import Constant, RandomNormal
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.callbacks import LearningRateScheduler
import h5py
from keras.utils import np_utils
from keras.applications import VGG16,VGG19
from keras.models import Model
import numpy as np
import cv2
import argparse
import sys
import tensorflow as tf
from keras import regularizers
import time
#specify the parameters


"""******************************************************************************************************************"""
def read_pairs(pairs_filename):
    '''this function mainly read the name pairs from a text file and return an array of names and numbers
    for example
    pairs = [[name1, pair_number_1, pair_ number_2],[name2, pair_number1, pair_number2],[...]]'''
    pairs = []
    path_list=[]
    issame_list=[]
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split() # Separates the elements when there is a space
            if pair[1]== '0':
                issame = 0
            elif pair[1] == '1':
                issame = 1
            elif pair[1] == '2':
                issame = 2
            elif pair[1] == '3':
                issame = 3
            else:
                print('Labels must be either 1 or 0')
                break
            path_list.append(pair[0])
            issame_list.append(issame)
            pairs.append(pair)
    return np.array(pairs), path_list, issame_list
"""******************************************************************************************************************"""
def video_data(filepath):
    video_matrix = []
    cap = cv2.VideoCapture(filepath)


    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-10
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    #print('length of the video = %g ---- height x width = %d x %d --- fps =%g' % (
     #   video_length, video_height, video_width, video_fps))
    counter = 1
    starting_point=5 #select the starting frame
    while (cap.isOpened()):

        ret, frame = cap.read()

        frame=cv2.resize(frame,(96,96),interpolation = cv2.INTER_AREA)
        if counter >= starting_point:
            video_matrix.append(frame)
        #if len(video_matrix)==frames:
        #    break
        #else:
        #    counter += 1
        if counter != video_length:
            #cv2.imshow('frame', frame)
            counter += 1
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cv2.destroyAllWindows()
    cap.release()


    return np.asarray((video_matrix)), len(video_matrix)



"""****************************************************************************************************************"""
def sample_people(dataset, people_per_batch, images_per_person,labels):
    """

    :param dataset: dataset from which to randomly select the videos
    :param people_per_batch: number of people per batch to sample or number of videos to sample from a list
    :param images_per_person:  #number of frames to be captured for each person
    :return: sampled images, and number of of images per class
    """
    nrof_images = people_per_batch * images_per_person # 24 * 1 = 24

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    # get the sequential array of number corresponding to the arrangment of files in the dataset
    class_indices = np.arange(nrof_classes) #
    # randomly shuffle the dataset
    np.random.shuffle(class_indices) #[3,1,5...,240]

    i = 0
    images_sampled = []
    num_per_class = []
    sampled_class_indices = []
    labels_sampled=[]
    # Sample images from these classes until we have enough
    while len(images_sampled) < nrof_images:

        class_index = class_indices[i]                       # ith video
        labels_sampled.append(labels[class_index])           # get the corresponding label for the ith video

        # get the image from the dataset
        imagedata = cv2.imread(dataset[class_index])
        images_sampled.append(imagedata) #contain images not path




        i += 1
    return images_sampled, labels_sampled
"""****************************************************************************************************************"""
def store_loss(file,loss):
    with open(file,'ab') as f:
        f.write(str(loss) + "\n")

"""****************************************************************************************************************"""
def main(args):
    """

    :param args:arguments to the function
    :return:
    """
    model = Sequential()
    '''adding a dropout to the input image'''

    model.add(Convolution2D(64,input_shape= (args.image_rows,args.image_cols,args.img_channels),
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=l2(args.l2_regul)))

    model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2),
                               padding='same'))

    model.add(Convolution2D(128,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=l2(args.l2_regul)))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='same'))

    model.add(Convolution2D(256, kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=l2(args.l2_regul)))
    model.add(Convolution2D(256, kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=l2(args.l2_regul)))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='same'))


    model.add(Convolution2D(512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=l2(args.l2_regul),))

    model.add(Convolution2D(512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=l2(args.l2_regul)))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='same'))

    model.add(Convolution2D(512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=l2(args.l2_regul)))

    model.add(Convolution2D(512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=l2(args.l2_regul)))

    model.add(MaxPooling2D(pool_size=(2, 2),
                          strides=(2, 2),
                          padding='same'))
    model.add(Flatten())


    with tf.device("/cpu:0"):
        model.add(Dense(4096, activation='relu',
                        kernel_regularizer=l2(args.l2_regul)))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu',
                        kernel_regularizer=l2(args.l2_regul)))
        model.add(Dropout(0.5))
        model.add(Dense(4,kernel_regularizer=l2(args.l2_regul),
                        activation='softmax'))


    '''*****************************************************************************************************************'''

    # compile the model (should be done *after* setting layers to non-trainable)

    #model.load_weights(
     #   '/home/yasar/Documents/facenet-research/facenet-master/facenet_recognition/tutorial5/log-loss/checkpoints/checkpoint' + str(
      #      450) + '.h5')
    lr = args.learning_rate
    optimizer = SGD(lr=lr,decay=1e-6, momentum=0.9)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

    pairs, train_data, issame_list = read_pairs(args.images_and_labels)
    test_pairs, test_data, test_issame_list = read_pairs(args.test_images_labels)

    batch_number=0

    while batch_number < args.max_epochs:

        average_loss = []

        starttime = time.time()

        for i in xrange(args.epoch_batch):


            train_images,train_labels=sample_people(train_data,args.batch_size,
                                                     args.images_per_person,issame_list)

            train_images=np.asarray(train_images)
            train_labels=np.asarray(train_labels).astype(np.int64)
            train_labels=np_utils.to_categorical(train_labels,4)



            training_loss = model.fit(train_images, train_labels,epochs=1,verbose=0)



            print('Epoch[%d] \t Sub-Epoch[%d]/[%d]: training_loss %g \t : labels: %s' %(batch_number,i,args.epoch_batch,training_loss.history['loss'][0],str(train_labels)))
            epoch_loss=[training_loss.history['loss'][0]]
            average_loss.append(epoch_loss)

        batch_number += 1

        average_tr_val_loss = np.sum(np.asarray(average_loss)) / args.epoch_batch

        #running validation

        test_images, test_labels = sample_people(test_data, args.batch_size,
                                                 args.images_per_person, test_issame_list)
        test_images = np.asarray(test_images)
        test_labels = np.asarray(test_labels).astype(np.int64)
        test_labels=np_utils.to_categorical(test_labels,4)

        testing_pred = model.predict(test_images,batch_size=12)
        correct_pred=np.equal(np.argmax(testing_pred,1),np.argmax(test_labels,1))
        training_accuracy=np.sum(correct_pred)/12

        Endtime = time.time() - starttime
        print("Validation Accuracy \t ", end='')
        print(training_accuracy)
        print ("Avg Training Loss: %g \t "  %(average_tr_val_loss))
        print("Total time taken %f" %Endtime)

        tloss=[average_tr_val_loss,training_accuracy]
        #store data to the text file
        store_loss(args.save_tr_val_loss, tloss)

        #learning rate schedule update
        if (batch_number == 200) | (batch_number == 400) | (batch_number == 450):
            model.save_weights(
                '/home/yasar/Documents/facenet-research/facenet-master/facenet_recognition/tutorial5/log-loss/checkpoints/checkpoint' + str(batch_number) + '.h5')

            lr *= 0.1

        model.save_weights(
            '/home/yasar/Documents/facenet-research/facenet-master/facenet_recognition/tutorial5/log-loss/checkpoints/checkpoint' + str(
                500) + '.h5')



"""******************************************************************************************************************"""
def parser_arguments(argv):
    parser=argparse.ArgumentParser()
    """Define the training directory"""
    parser.add_argument('--input_dir',type=str,
                        help='directory from where to get the training data',default='~/home/yaurehman2/facenet')

    parser.add_argument('--images_and_labels',type=str,
                        help='directory from where to get the training paths and ground truth',
                        default='/home/yasar/Documents/CASIA_FACE/train.txt')

    parser.add_argument('--test_images_labels',type=str,
                        help='direcotry where test iamges are stored ',
                        default='/home/yasar/Documents/CASIA_FACE/test.txt')

    parser.add_argument('--save_tr_val_loss',type=str,
                        help='saving the training and validation loss of first three layers',
                        default='/home/yasar/Documents/facenet-research/facenet-master/facenet_recognition/tutorial7/face_recognition_keras/face_VGG16/log-loss/log_loss')


    parser.add_argument('--save_all_loss', type=str,
                        help='saving the training and validation loss of first three layers',
                        default='/home/yaurehman2/Documents/Researchfirstyear/keras_research/face_recognition_keras/face_VGG16/log-loss/traing_loss')

    """**************************************************************************************************************"""

    """Specify the parameters for the VGG 16 CNN"""

    parser.add_argument('--batch_size',type=int,
                        help='input batch size to the network',default=12)


    parser.add_argument('--nb_classes',type=int,
                        help='number of classes for classification',default=2)

    parser.add_argument('--people_per_batch',type=int,
                        help='No of people per batch',default=12)

    parser.add_argument('--images_per_person',type=int,
                        help='no of images per person',default=1)

    parser.add_argument('--max_epochs',type=int,
                        help='maximum number of epochs for training',default=500)

    parser.add_argument('--epoch_batch',type=int,
                        help='Maximum epoch per batch per iteration',default=30)


    """**************************************************************************************************************"""


    parser.add_argument('--learning_rate',type=float,
                        help='Initial learning rate',default=0.000001)
    parser.add_argument('--l2_regul',type=float,
                        help='l2 regularizer',default=5e-4)
    parser.add_argument('--data_augmentation',type=str,
                        help='wheather to include data augmentation or not',default=False)
    parser.add_argument('--image_rows',type=int,
                        help='image height',default=150)
    parser.add_argument('--image_cols',type=int,
                        help='image width',default=150)
    parser.add_argument('--img_channels',type=int,
                        help='number of input channels in an image',default=3)
    parser.add_argument('--epoch_flag',type=int,
                        help='determine when to change the learning rate',default=1)


    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parser_arguments(sys.argv[1:]))
