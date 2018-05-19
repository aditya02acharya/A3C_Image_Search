# -*- coding: utf-8 -*-
"""
Created on Sat May 19 09:25:48 2018

@author: Aditya
"""

from GlobalConstants import MAX_ACTIONS, N_IMAGES
import tensorflow as tf
import os
import numpy as np
from time import sleep

from AC_Network import AC_Network
from Worker import Worker
from Environment import Environment
from helper import *

max_episode_length = 36
num_episodes = 10000
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = N_IMAGES # Observations are 6 * 6 matrix
a_size = MAX_ACTIONS # Agent can fixate on 36 possible locations and one stop action.
load_model = True
model_path = './Model'
episode_rewards = []
episode_lengths = []
episode_durations = []
episode_features = []
env = Environment()

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"): 
    master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
    saver = tf.train.Saver(max_to_keep=5)
    
with tf.Session() as sess:
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    
    for i in range(num_episodes):
        print(i)
        episode_buffer = []
        episode_values = []
        episode_frames = []
        episode_reward = 0
        episode_step_count = 0
        d = False
                
        s,f = env.reset()
        episode_frames.append(s)
        rnn_state = master_network.state_init
        batch_rnn_state = rnn_state
                
        while d == False:
            #Take an action using probabilities from policy network output.
            a_dist,v,rnn_state = sess.run([master_network.policy,master_network.value,master_network.state_out],
                                           feed_dict={master_network.inputs:[s],
                                           master_network.focus:[f],
                                           master_network.trainLength:1,
                                           master_network.state_in[0]:rnn_state[0],
                                           master_network.state_in[1]:rnn_state[1]})
                    
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)
                    
            s1, f1, r, d = env.test_step(a)

            episode_reward = r
            s = s1
            f = f1                    
            episode_step_count += 1
                    
            if d == True:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step_count-1) 
        episode_durations.append(env.total_time)
        episode_features.append(env.selected_feature)

m, low, up = mean_confidence_interval(episode_rewards)        
print("Mean utility : " + str(m) + ", " + str(low) + ", " + str(up))
m, low, up = mean_confidence_interval(episode_lengths)
print("Mean Fixations : " + str(m) + ", " + str(low) + ", " + str(up))
m, low, up = mean_confidence_interval(episode_durations)
print("Mean Duration : " + str(m) + ", " + str(low) + ", " + str(up))
m, low, up = mean_confidence_interval(episode_features)
print("Mean Features : " + str(m) + ", " + str(low) + ", " + str(up))
    