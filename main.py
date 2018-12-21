#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

LOGDIR = "LOGS"

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    with tf.name_scope(vgg_tag):

        # Default graph
        graph = tf.get_default_graph()

        # Load tensor from graph 
        input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
        keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        vgg_3_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        vgg_4_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        vgg_7_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

        # Add summary to tensorboard
        tf.summary.image('image_input', input_tensor)
        tf.summary.histogram('vgg_3', vgg_3_tensor)
        tf.summary.histogram('vgg_4', vgg_4_tensor)
        tf.summary.histogram('vgg_7', vgg_7_tensor)

    return input_tensor, keep_prob, vgg_3_tensor, vgg_4_tensor, vgg_7_tensor
tests.test_load_vgg(load_vgg, tf)

def conv_1x1(tensor, num_outputs, activation = None, name="conv_1x1"):
    """
    Wrapper funciton for 1x1 convolution with backbone tf.layers.conv2d
    tensor: 4D tensor
    num_outputs: output depth dimension (usually number of classes)
    """
    with tf.name_scope(name):
        conv_1x1_out = tf.layers.conv2d(tensor,
                                        num_outputs,
                                        kernel_size=(1,1),   # 1X1 conv
                                        strides=(1,1),
                                        padding='SAME',
                                        activation=activation,  
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        
        tf.summary.histogram(name, conv_1x1_out)

        return conv_1x1_out

def conv_transpose(tensor, num_outputs, kernel_size, strides, name='transpose_conv'):
    """
    Wrapper function for upsampling (transpose convolution operation) with backbone tf.layers.conv2d_transpose
    tensor: 4D tesnor
    num_outputs: output depth dimension (usually number of classes)
    """

    with tf.name_scope(name):
        transpose_conv_out = tf.layers.conv2d_transpose(tensor,
                                                        num_outputs,
                                                        kernel_size,
                                                        strides,
                                                        padding='SAME',
                                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        tf.summary.histogram(name, transpose_conv_out)

        return transpose_conv_out

def skip_connect(layer_1, layer_2, name='skip_connection'):
    """
    combine the specified layer_1 and layer_2.
    layer_1, layer_2: 4D tensors, and they need to be in the same dimension.
    retrun TF Operation
    """
    with tf.name_scope(name):
        skip_connection = tf.add(layer_1, layer_2)
        tf.summary.histogram(name, skip_connection)
        return skip_connection

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #vgg_layer3_out + 2x vgg_layer4_out + 4x vgg_layer7_out (pixel-wise addtion operation)
    
    # In the original paper, the vgg3 and vgg4 is upscaled
    vgg_4_upscaled = tf.multiply(vgg_layer4_out, 0.01)
    vgg_3_upscaled = tf.multiply(vgg_layer3_out, 0.001)

    # 1x1 convolution layer
    layer7_1x1 = conv_1x1(vgg_layer7_out, num_classes)
    layer4_1x1 = conv_1x1(vgg_4_upscaled, num_classes)
    layer3_1x1 = conv_1x1(vgg_3_upscaled, num_classes)
    
    # Upsampling the layer7, preparing for the skip connection with layer4
    layer7_2x = conv_transpose(layer7_1x1, num_classes, 4, 2)
    layer4_7_skip_connect = skip_connect(layer4_1x1, layer7_2x)

    # Upsampling the layer4_7_skip_connect, preparing for the skip connection with layer3
    layer4_2x = conv_transpose(layer4_7_skip_connect, num_classes, 4, 2)
    layer3_4_skip_connect = skip_connect(layer3_1x1, layer4_2x)

    # Upampling the final output to 8x
    output = conv_transpose(layer3_4_skip_connect, num_classes, 16, 8)

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes, reg_constant):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement 
    
    # Reshape 4D tensor to get logits as a 2D tensor where row: pixels, col:class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Add regularization term: https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = cross_entropy + reg_constant * sum(reg_losses)

    train_op = optimizer.minimize(loss=loss)

    return logits, train_op, loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, kp, lr, hyperparameter):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR + hyperparameter)
    writer.add_graph(sess.graph)

    counter = 1

    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):

            if(counter % 10 == 0):  # with summary
                _, loss, summary = sess.run([train_op, cross_entropy_loss, merged_summary],
                                            feed_dict={input_image: image,
                                                       correct_label: label,
                                                       keep_prob: kp,
                                                       learning_rate: lr})
                # summary must be written by writer, otherwise will have fetch 
                print("Epoch={}/{}, Loss={}".format(epoch+1, epochs, loss))
                writer.add_summary(summary, counter)
             
            else: # without summary
                _, loss = sess.run([train_op, cross_entropy_loss],
                                  feed_dict={input_image: image,
                                             correct_label: label,
                                             keep_prob: kp,
                                             learning_rate: lr})

            counter += 1

tests.test_train_nn(train_nn)

def make_hparam_string(learning_rate, keep_prob, l2_const):
    return "lr_{},kp_{},l2_{}".format(learning_rate, keep_prob, l2_const)

def run():
    epochs = 50
    batch_size = 16
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # CAREFUL: this removes ALL models and log results of the last run
    if tf.gfile.Exists(LOGDIR):
        tf.gfile.DeleteRecursively(LOGDIR)
    tf.gfile.MakeDirs(LOGDIR)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Grid search for the best combination of learning_rate, keep_prob, l2_reg
    for lr in [0.001, 0.0001, 0.00001]:
        for kp in [0.7, 0.8, 0.9]:
            for l2_const in [0.002, 0.005]:

                hparam = make_hparam_string(lr, kp, l2_const)
                print('Configuration {}'.format(hparam))
                model_path = LOGDIR + hparam + "/model"

                tf.reset_default_graph()
                with tf.Session() as sess:
                    # Path to vgg model
                    vgg_path = os.path.join(data_dir, 'vgg')
                    # Create function to get batches
                    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

                    # OPTIONAL: Augment Images for better results
                    #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

                    # TODO: Build NN using load_vgg, layers, and optimize function
                    learning_rate = tf.placeholder(tf.float32)
                    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])

                    input_image, keep_prob, vgg_3_tensor, vgg_4_tensor, vgg_7_tensor = load_vgg(sess, vgg_path)
                    last_layer = layers(vgg_3_tensor, vgg_4_tensor, vgg_7_tensor, num_classes)
                    logits, train_op, loss = optimize(last_layer, correct_label, learning_rate, num_classes, l2_const)

                    saver = tf.train.Saver()

                    # TODO: Train NN using the train_nn function
                    sess.run(tf.global_variables_initializer())
                    train_nn(sess, 
                            epochs,
                            batch_size,
                            get_batches_fn,
                            train_op,
                            loss,
                            input_image,
                            correct_label,
                            keep_prob,  # Placeholder
                            learning_rate, #Placeholder
                            kp, # Scalar
                            lr, # Scalar
                            hparam)

                    save_path = saver.save(sess, model_path)
                    print("Model is saved in path {}".format(save_path))

                    # TODO: Save inference data using helper.save_inference_samples
                    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

                    # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
