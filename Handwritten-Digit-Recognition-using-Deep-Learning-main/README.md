# Handwritten Digit Recognition Using Machine Learning and Deep Learning

## Published Paper 

[IJARCET-VOL-6-ISSUE-7-990-997](http://ijarcet.org/wp-content/uploads/IJARCET-VOL-6-ISSUE-7-990-997.pdf)

## Requirements

* Python 3.5+
* Scikit-Learn (latest version)
* Numpy (+ MKL for Windows)
* Matplotlib

## Usage

### Step 1: Download the MNIST Dataset

Download the MNIST dataset files using the following commands:

```sh
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

Alternatively, download the [dataset from here](https://github.com/anujdutt9/Handwritten-Digit-Recognition-using-Deep-Learning/blob/master/dataset.zip), unzip the files, and place them in the appropriate folders.

### Step 2: Organize Dataset Files

Unzip the files and place them in the `dataset` folder within the `MNIST_Dataset_Loader` directory under each ML algorithm folder:

```
KNN
|_ MNIST_Dataset_Loader
   |_ dataset
      |_ train-images-idx3-ubyte
      |_ train-labels-idx1-ubyte
      |_ t10k-images-idx3-ubyte
      |_ t10k-labels-idx1-ubyte
```

Repeat this process for the `SVM` and `RFC` folders.

### Step 3: Running the Code

Navigate to the directory of the algorithm you want to run and execute the corresponding Python file. For example, for K-Nearest Neighbors:

```sh
cd 1. K Nearest Neighbors/
python knn.py
```

Or use Python 3:

```sh
python3 knn.py
```

This will execute the code and log all print statements into the `summary.log` file. To view output in the command prompt, comment out lines 16, 17, 18, 106, and 107 in the script.

You can also run the scripts using an IDE like PyCharm.

Repeat the steps for the `SVM` and `RFC` algorithms.

### Step 4: Running the CNN Code

The CNN code will automatically download the MNIST dataset. Run the file with:

```sh
python CNN_MNIST.py
```

Or with Python 3:

```sh
python3 CNN_MNIST.py
```

### Step 5: Saving the CNN Model Weights

To save the model weights after training, use:

```sh
python CNN_MNIST.py --save_model 1 --save_weights cnn_weights.hdf5
```

Or with Python 3:

```sh
python3 CNN_MNIST.py --save_model 1 --save_weights cnn_weights.hdf5
```

### Step 6: Loading Saved Model Weights

To load previously saved model weights and skip training, use:

```sh
python CNN_MNIST.py --load_model 1 --save_weights cnn_weights.hdf5
```

Or with Python 3:

```sh
python3 CNN_MNIST.py --load_model 1 --save_weights cnn_weights.hdf5
```

## Accuracy using Machine Learning Algorithms:

- **K Nearest Neighbors:** 96.67%
- **SVM:** 97.91%
- **Random Forest Classifier:** 96.82%

## Accuracy using Deep Neural Networks:

- **Three Layer Convolutional Neural Network using TensorFlow:** 99.70%
- **Three Layer Convolutional Neural Network using Keras and Theano:** 98.75%

*All code is written in Python 3.5 and executed on an Intel Xeon Processor/AWS EC2 Server.*

## Video Link

[Watch the video](https://www.youtube.com/watch?v=7kpYpmw5FfE)

## Test Images Classification Output

![Output a1](Outputs/output.png "Output a1")
