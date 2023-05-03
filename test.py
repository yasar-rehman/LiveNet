
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

from keras.utils import np_utils
from keras.applications import VGG16,VGG19
from keras.models import Model
import numpy as np
import cv2
import argparse
import sys
import tensorflow as tf
from keras import regularizers
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
            pair = line.strip().split()  # Separates the elements when there is a space
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




def store_loss(file,loss):
    with open(file,'ab') as f:
        f.write(str(loss) + "\n")

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

"""****************************************************************************************************************"""
def main(args):
    """

    :param args:arguments to the function
    :return:
    """
    model = Sequential()




    '''adding a dropout to the input image'''

    model.add(Convolution2D(64,input_shape= (args.image_rows,args.image_cols,args.img_channels),
                            kernel_size=(7, 7),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=l2(args.l2_regul)))

    model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2),
                               padding='same'))

    model.add(Convolution2D(128,
                            kernel_size=(5, 5),
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
    #model.add(Dropout(0.5))

    '''Second Convolution Layer'''

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
        model.add(Dense(4096,activation='relu',
                        kernel_regularizer=l2(args.l2_regul)))
        #model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu',
                        kernel_regularizer=l2(args.l2_regul)))
        #model.add(Dropout(0.5))
        model.add(Dense(4,kernel_regularizer=l2(args.l2_regul),
                        activation='softmax'))


    '''*****************************************************************************************************************'''

    '''classification and softmax'''


    """load a pretrained VGG-16 model"""

    # compile the model (should be done *after* setting layers to non-trainable)
    #lr = args.learning_rate
    #optimizer = SGD(lr=lr,decay=1e-6, momentum=0.9)

    #model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')


    test_pairs, test_data, test_issame_list = read_pairs(args.images_and_labels)

    batch_number=0
    model.load_weights(
        '/face_recognition_keras/face_VGG16/checkpoints500.h5')
    accuracy=[]
    temp_accuracy=[]
    FAR = []
    temp_FAR=[]
    FRR = []
    temp_FRR=[]


    for i in range(len(test_data)):
        frames = cv2.imread(test_data[i])
        frames = prewhiten(frames)
        frames = cv2.resize(frames,(96,96),interpolation=cv2.INTER_AREA)
        labels=[test_issame_list[i]]

        #computing the accuracy
        test_images = np.reshape(np.asarray(frames),(1,96,96,3))
        test_labels = np.reshape(np.asarray(labels).astype(np.int64),(-1,))
        prediction=model.predict(test_images)
        #print (prediction)
        acc, tr,far,frr=livness_FAR(prediction,test_labels)

        #v=np.argmax(prediction)
        #correct_prediction=np.equal(np.argmax(prediction),test_labels)

        temp_accuracy.append(tr)
        temp_FAR.append(far)
        temp_FRR.append(frr)

        if ((i+1)%100 == 0) & (i!=0):
            corr1=np.sum(np.asarray(temp_accuracy))/100
            far1 = np.sum(np.asarray(temp_FAR)) / 100
            frr1 = np.sum(np.asarray(temp_FRR)) / 100

            accuracy.append(corr1)
            FAR.append(far1)
            FRR.append(frr1)
            print (" Validdation accuracy : %f\t False Accept: %f \t False Reject %f " %(corr1,far1,frr1))
            temp_accuracy = []
            temp_FAR = []
            temp_FRR = []
            metrics=[corr1,far1,frr1]
            store_loss(args.save_all_factors,metrics)
    final_accuracy=np.sum(np.asarray(accuracy)) / (len(test_data)/100)
    final_far = np.sum(np.asarray(FAR)) / (len(test_data) / 100)
    final_frr = np.sum(np.asarray(FRR)) / (len(test_data) / 100)

    print ("final Accuracy: %f FAR: %f FRR: %f "%(final_accuracy,final_far,final_frr))

        #tloss=[average_tr_val_loss,testing_loss]
        #store data to the text file
        #store_loss(args.save_tr_val_loss, tloss)

        #learning rate schedule update
        #if (batch_number == 50) | (batch_number == 100) | (batch_number == 150):
         #   lr *= 0.1



def livness_FAR(pre_embeddings,labels):

    vv=np.argmax(pre_embeddings,axis=1) # get the pre-embeddings
    correct_prediction = np.equal(vv,labels) # get the corresponding labels
    # convert a multi-class classsification to a binary class classification problem
    # convert the prediction to a binary
    if vv > 0:
        vv = np.array([0])
    else:
        vv=np.array([1])
    # convert the labels into binary
    if labels > 0:
        labels = np.array([0]) # The image has been considered as fake
    else:
        labels = np.array([1]) # The image has been considered as true

        # real face label = 1
        # fake face label = 0

    #print ("prediction",end ='')
    #print(vv.astype(np.bool),end='')  # print the  prediction as logical values
    #print("\t correct",end='')
    #print(labels.astype(np.bool))  # print the labels as logical values
    #print (vv.astype(np.bool))


    accuracy=correct_prediction
    #print ("accuracy %f" %accuracy)
    true_accept = np.logical_not(np.logical_xor(vv.astype(np.bool),labels.astype(np.bool))) # prediction == labels

    false_accept = np.logical_and(vv.astype(np.bool),np.logical_not(labels.astype(np.bool))) # prediction = live, label = Fake

    false_reject=np.logical_and(np.logical_not(vv.astype(np.bool)),labels.astype(np.bool)) # prediction = fake, label = live

    #print ("True_accept %g: \t FAR %g: \t FRR %g \t" %(true_accept,false_accept,false_reject))

    #print("***********************************************")

    #false_accept=np.sum(np.logical_and(correct_prediction,test_issame))/pre_embeddings.shape[0]
    return accuracy, true_accept,false_accept,false_reject



"""******************************************************************************************************************"""
def parser_arguments(argv):
    parser=argparse.ArgumentParser()
    """Define the training directory"""
    parser.add_argument('--input_dir',type=str,
                        help='directory from where to get the training data',default='~/home/yaurehman2/facenet')

    parser.add_argument('--images_and_labels',type=str,
                        help='directory from where to get the training paths and ground truth',
                        default='/CASIA_FACE/train.txt')

    parser.add_argument('--test_images_labels',type=str,
                        help='direcotry where test iamges are stored ',
                        default='/CASIA_FACE/test.txt')


    parser.add_argument('--save_all_factors', type=str,
                        help='saving the training and validation loss of first three layers',
                        default='/face_recognition_keras/face_VGG16/log-loss/metrics1')

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
                        help='maximum number of epochs for training',default=200)

    parser.add_argument('--epoch_batch',type=int,
                        help='Maximum epoch per batch per iteration',default=30)


    """**************************************************************************************************************"""


    parser.add_argument('--learning_rate',type=float,
                        help='Initial learning rate',default=0.01)
    parser.add_argument('--l2_regul',type=float,
                        help='l2 regularizer',default=0)
    parser.add_argument('--data_augmentation',type=str,
                        help='wheather to include data augmentation or not',default=False)
    parser.add_argument('--image_rows',type=int,
                        help='image height',default=96)
    parser.add_argument('--image_cols',type=int,
                        help='image width',default=96)
    parser.add_argument('--img_channels',type=int,
                        help='number of input channels in an image',default=3)
    parser.add_argument('--epoch_flag',type=int,
                        help='determine when to change the learning rate',default=1)


    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parser_arguments(sys.argv[1:]))
