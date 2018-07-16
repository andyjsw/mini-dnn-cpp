#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include "src/layer/fully_connected.h"
#include "src/mnist.h"
#include "src/mse_loss.h"
#include "src/layer/sigmoid.h"

int main()
{
  // data
  MNIST dataset("../data/mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  const int batch_size = 32;
  std::cout << "mnist: " << n_train << std::endl;
  std::cout << dataset.test_labels.cols() << std::endl;
  // dnn
  FullyConnected fc1(dim_in, 128);
  Sigmoid sigmoid1;
  FullyConnected fc2(128, 10);
  Sigmoid sigmoid2;
  // loss
  MSE mse;
  SGD opt(0.01, 0.0001);
  const int n_epoch = 5;
  for (int epoch = 0; epoch < n_epoch; epoch ++) {
    for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
      Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in, 
                                        std::min(batch_size, n_train - start_idx));
      Matrix label_batch = dataset.train_labels.block(0, start_idx, 1, 
                                        std::min(batch_size, n_train - start_idx));
      Matrix target_batch = one_hot_encode(label_batch, 10);
      fc1.forward(x_batch);
      sigmoid1.forward(fc1.output());
      fc2.forward(sigmoid1.output());
      sigmoid2.forward(fc2.output());
      mse.evaluate(sigmoid2.output(), target_batch);
      
      int ith_batch = start_idx / batch_size;
      if((ith_batch % 10) == 0){
        std::cout << ith_batch << "-th batch, loss: " << mse.output() << std::endl;
      }

      sigmoid2.backward(fc2.output(), mse.back_gradient());
      fc2.backward(sigmoid1.output(), sigmoid2.back_gradient());
      sigmoid1.backward(fc1.output(), fc2.back_gradient());
      fc1.backward(x_batch, sigmoid1.back_gradient());
      fc2.update(opt);
      fc1.update(opt);
    }
    // test
    fc1.forward(dataset.test_data);
    sigmoid1.forward(fc1.output());
    fc2.forward(sigmoid1.output());
    sigmoid2.forward(fc2.output());
    float acc = compute_accuracy(sigmoid2.output(), dataset.test_labels);
    std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
  }
  return 0;
}

