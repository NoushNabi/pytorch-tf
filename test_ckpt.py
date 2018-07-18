import tensorflow as tf
import numpy as np
from squeezenet import SqueezeNet

sess = tf.Session()

model = SqueezeNet()

saver = tf.train.Saver()
saver.restore(sess, "./ckpt/squeezenet.ckpt")
input_image = model.image
classifier = model.classifier
features = model.features

from PIL import Image
from scipy.misc import imresize
import os

with open('labels.txt') as fp:
    labels = [c[:-2].split(':')[1] for c in fp.readlines()]
def get_img(filename):
    vec = np.array(Image.open(filename))
    vec = imresize(vec,(224,224)).astype(np.float32)/255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    vec = (vec-mean)/std
    return vec
    
img_dir = './images/'
img_names = [x for x in os.listdir(img_dir) if 'jpeg' in x.lower()]
imgs = [get_img(os.path.join(img_dir,x)) for x in img_names]

scores = sess.run(classifier,feed_dict={input_image:np.array(imgs).reshape([-1,224,224,3])})
for idx,s in enumerate(np.argmax(scores,1)):
    print(img_names[idx],labels[s])