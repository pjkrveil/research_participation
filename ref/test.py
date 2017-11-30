import train_model
from model_vgg16 import Vggmodel
import tensorflow as tf
import time

vgg = Vggmodel(optimizer=tf.train.GradientDescentOptimizer, fine_tuning=True, lr=0.001, dropout=True, adaptive_ratio=1.0)
train_model.train(vgg, timestamp=int(time.time()))
