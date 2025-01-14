#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"
#include "src/layer/kernels/kernels.h"

int main()
{
    cuda_helper cuda_helper;
    cuda_helper.print_device_info();

    MNIST dataset("../data/fashion/");
    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();

    std::cout << "Fashion-mnist training samples: " << n_train << std::endl;
    std::cout << "Fashion-mnist test samples: " << dataset.test_labels.cols() << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

    float acc = 0;
    std::cout << "Parallel forward propagation" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    Network dnn;
    Layer *conv1 = new Conv(1, 28, 28, 6, 5, 5);
    Layer *pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer *conv2 = new Conv(6, 12, 12, 16, 5, 5);
    Layer *pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
    Layer *fc1 = new FullyConnected(pool2->output_dim(), 120);
    Layer *fc2 = new FullyConnected(120, 84);
    Layer *fc3 = new FullyConnected(84, 10);
    Layer *relu_conv1 = new ReLU;
    Layer *relu_conv2 = new ReLU;
    Layer *relu_fc1 = new ReLU;
    Layer *relu_fc2 = new ReLU;
    Layer *softmax = new Softmax;
    dnn.add_layer(conv1);
    dnn.add_layer(relu_conv1);
    dnn.add_layer(pool1);
    dnn.add_layer(conv2);
    dnn.add_layer(relu_conv2);
    dnn.add_layer(pool2);
    dnn.add_layer(fc1);
    dnn.add_layer(relu_fc1);
    dnn.add_layer(fc2);
    dnn.add_layer(relu_fc2);
    dnn.add_layer(fc3);
    dnn.add_layer(softmax);
    Loss *loss = new CrossEntropy;
    dnn.add_loss(loss);
    // Load parameters
    dnn.load_parameters("../model/params-10eps.txt");

    dnn.forward(dataset.test_data);
    acc = compute_accuracy(dnn.output(), dataset.test_labels);
    std::cout << "Accuracy: " << acc << std::endl;
    std::cout << "-----------------------------------------" << std::endl;


    return 0;
}