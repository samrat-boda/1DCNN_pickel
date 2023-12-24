import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 
tf.compat.v1.disable_eager_execution()
import scipy.io as scio 
import scipy.io as sio
import pickle
from pathlib import Path
import os
import scipy  
from tensorflow.python.framework import ops  
from matplotlib.colors import ListedColormap   
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report 

import pickle
def load_parameters(file_path):
    with open(file_path, 'rb') as file:
        loaded_parameters = pickle.load(file)
    return loaded_parameters

# Load parameters from the saved file
loaded_parameters = load_parameters('1DCNN_Cuprite_parameters.pkl')


def mynetwork(x, parameters, isTraining, momentums = 0.9): 
    print(x.shape)
    x = tf.reshape(x, [-1, 1, 1, 224], name = "x")  
    print(x.shape)
    with tf.compat.v1.name_scope("x_layer_1"):                                    
         
         x_z1 = tf.nn.conv2d(x, filters=parameters['x_w1'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x_b1']  
         x_z1_bn = tf.compat.v1.layers.batch_normalization(x_z1, momentum = momentums, training = isTraining)   
         x_a1 = tf.nn.relu(x_z1_bn)
         
    with tf.compat.v1.name_scope("x_layer_3"):
        
         x_z2 = tf.nn.conv2d(x_a1, filters=parameters['x_w2'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x_b2'] 
         x_z2_shape = x_z2.get_shape().as_list() 
         x_z2_2d = tf.reshape(x_z2, [-1, x_z2_shape[1] * x_z2_shape[2] * x_z2_shape[3]])   
                    
         
    l2_loss =   tf.nn.l2_loss(parameters['x_w1']) + tf.nn.l2_loss(parameters['x_w2'])
               
    return x_z2_2d, l2_loss


input_data = ...  # Your input data
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 224], name="input_placeholder")
# tf.compat.v1.placeholder(tf.float32, [None, n_x], name = "x")  

Cuprite_data = scipy.io.loadmat('Cuprite.mat')
Cuprite_data =Cuprite_data['X'] 
# print(Cuprite_data.shape)
Cuprite_data= Cuprite_data.reshape(-1, Cuprite_data.shape[-1]) 
maxVal=np.amax(Cuprite_data)
minVal=np.amin(Cuprite_data) 
Cuprite_shifted = Cuprite_data + abs(minVal) 
Cuprite_norm = Cuprite_shifted / (abs(minVal) + abs(maxVal))
Cuprite_norm = Cuprite_norm + 1e-6  

isTraining = tf.compat.v1.placeholder(tf.bool, name="is_training_placeholder")
momentum_value = 0.9  # You can adjust this value if needed
output, l2_loss = mynetwork(x, loaded_parameters, isTraining, momentums=momentum_value)


with tf.compat.v1.Session() as sess:
    # Initialize variables
    sess.run(tf.compat.v1.global_variables_initializer())

    # Feed input data and isTraining value
    feed_dict = {x: Cuprite_norm, isTraining: False}  # Set isTraining to True during training

    # Get the output
    output_result, l2_loss_result = sess.run([output, l2_loss], feed_dict=feed_dict)

    # Print or use the output as needed
    print("Output:", output_result.shape)
    print("L2 Loss:", l2_loss_result)

num_labels = 12
output_ = np.argmax(output_result, axis=1)
output_ = output_.reshape(-1, 1)  


       # Reshape label array to 2D array
output_2d = np.reshape(output_, (512, 614))

        # Define color map for elements
colors = ['#8B0000', '#FFA500', '#FFFF00', '#00CED1', '#228B22', '#008000', '#808080', '#FFC0CB', '#800080', '#FF69B4', '#0000FF', '#a52a2a']
        # colors = ['#660000', '#CC6600', '#CCCC00', '#006666', '#004400', '#003300', '#555555', '#FF99CC', '#660066', '#FF3399', '#000066', '#660000']
cmap = ListedColormap(colors)

        # Define color for unlabeled pixels
cmap.set_under('white')

        # Plot the image with labels
fig, ax = plt.subplots()
im = ax.imshow(output_2d, cmap=cmap, vmin=0.5, vmax=12.5, interpolation='nearest')
cbar = fig.colorbar(im, ax=ax, ticks=range(1, 13))
cbar.ax.set_yticklabels(["#1 Alunite", "#2 Andradite", "#3 Buddingtonite", "#4 Dumortierite", "#5 Kaolinite1","#6 Kaolinite2", "#7 Muscovite", "#8 Montmorillonite", "#9 Nontronite", "#10 Pyrope","#11 Sphene", "#12 Chalcedony"])
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_ylabel('Mineral Labels', fontsize=10)
plt.title("Cuprite Data Mineral Classification Using 1-DCNN")
plt.show()

