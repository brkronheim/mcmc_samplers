# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:46:09 2021

@author: brade
"""

import tensorflow as tf

x = tf.ones((32))
print(tf.reshape(tf.repeat(x,4),(32,2,2)))