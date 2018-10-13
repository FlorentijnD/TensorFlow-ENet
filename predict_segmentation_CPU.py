import tensorflow as tf
import os, sys
from enet import ENet, ENet_arg_scope
from preprocessing import preprocess
from scipy.misc import imsave
import numpy as np, time
from PIL import Image
slim = tf.contrib.slim

#Create the photo directory
checkpoint_dir = "./checkpoint"
photo_dir = checkpoint_dir + "/test_images"
if not os.path.exists(photo_dir):
    os.mkdir(photo_dir)

image = Image.open("dataset/test/0001TP_008550.png")
seg_image = Image.open("dataset/color_palette.png")
palette = seg_image.getpalette()

# resize image
base_width  = image.size[0]
base_height = image.size[1]
image = image.resize((480, 360), Image.ANTIALIAS)

# delete alpha channel
if image.mode == "RGBA":
    image = image.convert("RGB")

# normalization
image = np.asarray(image)
prepimg = image / 255.0



with tf.Graph().as_default():

    # Read .pb file
    with tf.gfile.FastGFile("checkpoint/semanticsegmentation_enet.pb", "rb") as f:
        graphdef = tf.GraphDef()
        graphdef.ParseFromString(f.read())
        _ = tf.import_graph_def(graphdef, name="")
    sess = tf.Session()

    # Segmentation
    t1 = time.time()
    outputs = sess.run("ENet/logits_to_softmax:0", {"input:0":[prepimg]})
    print("elapsedtime =", time.time() - t1)

    # View
    # output_shapes = (1, 360, 480, 3)
    output = outputs[0]
    res = np.argmax(output, axis=2)
    image = Image.fromarray(np.uint8(res), mode="P")
    image.putpalette(palette)
    image = image.convert("RGB")
    image.save(photo_dir + "/segmented_image.png")
    print("finish!!")
