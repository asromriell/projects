__author__ = 'davidwen'

import glob
import tensorflow as tf
from skimage import io


# http://stackoverflow.com/questions/33648322/tensorflow-image-reading-display

#filename_queue = tf.train.string_input_producer(['/Users/davidwen/images_emotion/03-jaffe-japanese/H037370.png']) #  list of files to read

path = "/Users/davidwen/images_emotion/03-jaffe-japanese/"
files = glob.glob(path + "/A0372*png")
filename_queue = tf.train.string_input_producer(files)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_png(value)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)

    # Start populating the filename queue.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(len(files)): #length of your filename list
        image = my_img.eval()

        print(image.shape)
        #io.imsave("test.png", image)

    coord.request_stop()
    coord.join(threads)


