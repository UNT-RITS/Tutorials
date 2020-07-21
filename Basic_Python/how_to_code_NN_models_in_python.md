# How to code Neural Network models in Python

## :construction: ... Work in Progress ... :construction:

---------------------

Content:

[What is an Artificial Neural Network?](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/how_to_code_NN_models_in_python.md#what-is-an-artificial-neural-network)

[What are the main components and why do we need each of them?](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/how_to_code_NN_models_in_python.md#what-are-the-main-components-and-why-do-we-need-each-of-them)
        
- Weights, Bias and Layers

- Activation Function: **Linear Activation Function** and **Non-linear Activation Function** (_Sigmoid, Tanh_ & _ReLU_)

- Derivatives

[Architectures of Artificial Neural Network](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/how_to_code_NN_models_in_python.md#architectures-of-artificial-neural-network)
    
[What is Machine Learning?](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/how_to_code_NN_models_in_python.md#what-is-machine-learning)

[Machine Learning Categories by the level of human supervision:]()
  1. [Supervised Learning](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/how_to_code_NN_models_in_python.md#1-supervised-learning)
  2. [Unsupervised Learning](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/how_to_code_NN_models_in_python.md#2unsupervised-learning)
  3. [Semi-Supervised Learning](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/how_to_code_NN_models_in_python.md#3semi-supervised-learning)
  4. [Self-Supervised Learning](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/how_to_code_NN_models_in_python.md#4self-supervised-learning)
  5. [Reinforcement Learning](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/how_to_code_NN_models_in_python.md#5reinforcement-learning-rl)

---------------------

## What is an Artificial Neural Network?
**Artificial Neural networks (ANN)** are a set of algorithms, modeled in a similar way the human brain works, developed to recognize and predict patterns. They interpret given data through a machine perception, using labeling or collecting raw input. The patterns they recognize are numerical, expressed as vectors, and so is the output before having assigned a meaning (check this [video](https://www.youtube.com/watch?v=aircAruvnKk) explanation). 
Therefore, it is essential to convert the real-world input data, like images, sounds or text, into numerical values.

Most of the existing neural networks architectures are shown in the following picture:
![ANNs](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/ann1.png)

Deep learning is the name we use for _“stratified neural networks”_ = **networks composed of several layers**.

The layers are represented by a column vector and the elements of the vector can be thought of as _nodes_ (also named _neurons_ or _units_).

A **node (neuron)** is just **a place where computation happens**. It receives input from some other nodes, or from an external source and computes the outcome. Each input has an associated **weight (w)**, which is assigned on the basis of its relative importance to other inputs. Then, the output is computed by combining a set of **coefficients**, or **weights**, that either _amplify_ or _reduce_ that input depending on its importance. 

The input-weights are multiplied between each other and summed up. The sum is passed through a node’s so-called **activation function**, to determine whether or to what extent that signal should progress further through the network to affect the ultimate outcome. If the signal passes through, the neuron has been **activated**.

## What are the main components and why do we need each of them?

### 1. Why do we need Weights, Bias and Layers?

**Weight** shows the strength of the particular node. In other words, the weight is the assigned significance of an input in comparison with the relative importance of other inputs.

A **bias** value allows you to shift the activation function curve up or down.

A neural network can usually consist of three types of nodes:

   - **Input Nodes** – they provide information from the outside world to the network and are referred to as the “Input Layer”. No computation is performed in any of the Input nodes – they just pass on the information to the hidden nodes.

   - **Hidden Nodes** – they have no direct connection with the outside world and form a so called “Hidden Layer”. They perform computations and transfer information from the input nodes to the output nodes.

   - **Output Nodes** – they are collectively referred to as the “Output Layer” and are responsible for computations and mapping information from the network to the outside world.

### 2. Why do we need Activation Function?

Also known as Transfer Function, it is used to determine the output of neural network like yes or no. It maps the resulting values in between (0, 1) or (-1, 1) etc. (depending upon the type of function).

The Activation Functions can be basically divided into 2 types:

a. **Linear Activation Function**

The function is a line or linear. Therefore, the output of the function will not be restricted between any range.

       Equation: f(x) = x.              Range: (-∞,∞).

Not helpful with the complexity of data that is fed to the neural networks. Doesn’t matter how many neurons we link together the behavior will be the same.

b. **Non-Linear Activation Function**

The Nonlinear Activation Functions are the most used activation functions. They make it easy for the model to generalize and adapt to the variations of the data and to better categorize the output.

The most common non-linear activation functions are:

- Sigmoid or Logistic Activation Function 
- Tanh or hyperbolic tangent Activation Function
- ReLU (Rectified Linear Unit) Activation Function 

### 3. Why the derivative/differentiation is being used?

We use differentiation in almost every part of Machine Learning and Deep Learning, because when updating the curve, we need to know in which direction and how much to change or update the curve depending upon the slope. 

In the following table it is a clear distinction and classification of some functions and their derivates.

![ANNs](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/ann2.png)

## Architectures of Artificial Neural Network

# Artificial Neural Networks Architectures

# :construction: ... Work in Progress ... :construction:

Content

- [Introduction]()
- [Feedforward Neural Networks (FF)]():
  - [Singe-layer Perceptron (Perceptron)]()
  - [Multi-Layer Perceptron (MLP)]()
- [Radial Basis Function (RBF) Neural Network ]()
- [Deep Feedforward Network (DFF)]()
- [Recurrent Networks]():
  - [Recurrent Neural Networks (RNN)]()
  - [Long-Short Term Memory (LSTM) Neural Networks]()
  - [Gated Recurrent Unit (GRU) Neural Networks]()
  
--------------------

## Introduction

The neural network architecture represents the way in which the elements of the network are binned together. The architecture will define the behavior of the network. Considering their architecture, Artificial Neural Networks can be classified as follows:

# 1.	Feedforward Neural Networks (FF)

Feedforward Neural Networks were the first type of Artificial Neural Network. These kinds of nets have a large variance but there are mainly two pioneers:

## 1.1.	Singe-layer Perceptron (Perceptron) - [Coursera](https://www.coursera.org/lecture/mind-machine-computational-vision/perceptrons-4AT1O)

The **perceptron** is the simplest type of feedforward neural network, compost of only one neuron, where:  

  -	It takes some inputs, <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/xi.png" alt="inpits" width="25" height="25">, and each of them is multiplied with their related weight, <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/wi.png" alt="inpits" width="25" height="25"> : 

 ![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann3.png)

  - Then, it sums them up: <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/sum_n.png" alt="inpits" width="50" height="50">, particularly for the example above, we will have: <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/sum_5.png" alt="inpits" width="50" height="50">.

  -	The **activation function** (usually this is a _logistic function_) squashes the above summation and passes it to the output layer.

**Where do we use Perceptron?**

Perceptron is good for classification and decision-making systems because of the logistic function properties. That being the case, it is usually used to classify the data into two parts and it is also known as a Linear Binary Classifier. 

The issue with individual neurons rapidly arises when trying to solve every day - life problems due to the real-world complexity. Determined to address this challenge, researchers observed that by combining neurons together, their decision is basically combined getting as a result insight from more elaborated data. 

And so, the creation of an Artificial Neural Network with more neurons seems to be the answer.

## 1.2.	Multi-Layer Perceptron (MLP) - [Coursera](https://www.coursera.org/lecture/intro-to-deep-learning/multilayer-perceptron-mlp-yy1NV)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann4.png" width="250" height="200" > 

The **multi-layer perceptron** is a type of feedforward neural network that introduces the _multiple neurons_ design, where:

  - all nodes are fully connected (each node connects all the nodes from the next layer);
  - information only travels forward in the network (no loops);
  - there is one hidden layer (if present).

**Where do we use Multi-Layer Perceptron?**

Contrasting single layer perceptrons, MLPs are capable to manipulate non-linearly separable data. 

Therefore, these nets are used in many applications, but not by themselves. Most of the time they stand as a pillar of support in the construction of the next neural networks’ architecture. 

# 2.	Radial Basis Function (RBF) Neural Network 

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann5.png" width="280" height="180" > 

**Radial Basis Function Networks** are feedforward nets _with a different activation function_ in place of the logistic function, named the **radial basis function (RBF)**. 

Intuitively, RBF answers the question _“How far is the target from where we are?”_. More technically, it is a real-valued function defined as the difference between the input and some fixed point, called the **center**. 

The RBF chosen is usually a Gaussian, and it behaves like in the following example: 

<img src="https://www.digitalvidya.com/wp-content/uploads/2019/01/Image-3.gif" width="300" height="200" > 

RBN is strictly limited to have **exactly one hidden layer** (green dots in the related figure). Here, this hidden layer is known as a **feature vector**.

Moreover, Wikipedia says:

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann5-1.png)

**Where do we use Radial Basis Function Network?**

RBF nets are used in [function approximation](https://github.com/andrewdyates/Radial-Basis-Function-Neural-Network/blob/master/CSE779LabReport2.pdf) ([paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.6019&rep=rep1&type=pdf) & [code](https://github.com/thomberg1/UniversalFunctionApproximation)), time series prediction ([paper](https://dl.acm.org/doi/pdf/10.1145/3305160.3305187) & [code](https://www.mathworks.com/matlabcentral/fileexchange/66216-mackey-glass-time-series-prediction-using-radial-basis-function-rbf-neural-network)), and machine/system control (for example as a replacement of Partial Integral Derivative controllers). 

# 3.	Deep Feedforward Network (DFF) - [Coursera](https://www.coursera.org/lecture/ai/deep-feed-forward-neural-networks-kfTED)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann6.png" width="200" height="180" > 

**Deep Feedforward Neural Networks** are feedforward nets with _more than one hidden layer_.

It follows the rules:
  - all nodes are fully connected;
  - activation flows from input layer to output, without back loops;
  - there is **more than one layer** between input and output (hidden layers – green dots).

When training the traditional FF model, only a small amount of error information passes to the next layer. With more layers, DFF is able to learn more about errors; however, it becomes impractical as the amount of training time required increases.

Nowadays, a series of effective methods for training DFF have been developed, which have formed the core of modern machine learning systems and enable the functionality of feedforward neural networks.

**Where do we use Deep Feed-Forward Network?**

DFF is being used for automatic language identification (the process of automatically identifying the language spoken or written: [paper](https://static.googleusercontent.com/media/research.google.com/en/pubs/archive/42538.pdf), [paper](https://arxiv.org/pdf/1708.04811.pdf) & [code](https://github.com/HPI-DeepLearning/crnn-lid)), acoustic modeling for speech recognition ([thesis](https://mi.eng.cam.ac.uk/~mjfg/thesis_cw564.pdf), [paper](https://arxiv.org/pdf/1809.02108.pdf) & [code](https://github.com/amitai1992/AutomatedLipReading)), and other.

# 4.	Recurrent Networks 

## 4.1. Recurrent Neural Networks (RNN) – [Coursera](https://www.coursera.org/lecture/nlp-sequence-models/recurrent-neural-network-model-ftkzt) 

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann7.png" width="280" height="180" > 

The **Recurrent Neural Networks (RNN)** introduce the _recurrent cells_, a special type of cells located in the hidden layer (blue dots) and responsible of receiving its own output with a fixed delay — for one or more iterations creating loops. Apart from that, this network is like a usual FF net and so it follows similar rules:

  - all nodes are fully connected;
  - activation flows from input layer to output, with back loops;
  - there is more than one layer between input and output (hidden layers).

**Where do we use Recurrent Neural Networks?**

RNN is mainly used when the context is important — when decisions from past iterations or samples can influence the current ones, such as sentiment analysis ([paper](https://arxiv.org/pdf/1902.09314v2.pdf) & [code](https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/SentimentAnalysisProject)).

The most common examples of such contexts are texts — a word can be analyzed only in the context of previous words or sentences. RNNs can process texts by “keeping in mind” ten previous words.

[More on RNN](https://github.com/kjw0612/awesome-rnn) - A curated list of resources dedicated to recurrent neural networks.

## 4.2.	Long-Short Term Memory (LSTM) Neural Networks - [Coursera](https://www.coursera.org/lecture/tensorflow-sequences-time-series-and-prediction/lstm-5Iebr)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann8.png" width="280" height="180" > 

**LSTM** is a subcategory of RNN. They introduce a _memory cell_ – a special cell that can store and recall facts from time dependent data sequences. (tutorial like [paper](https://arxiv.org/pdf/1909.09586.pdf))

Memory cells are actually composed of a couple of elements — called gates, that are recurrent and control how information is being remembered and forgotten. 

  -	The input Gate determines how much of the last sample is stored in memory; 
  -	The output gate adjusts the amount of data transferred to the next level; 
  -	The Forget Gate controls the rate at which memory is stored.

Note that there are **no activation functions** between blocks.

**Where do we use Long-Short Term Memory Network?**

LSTM networks are used when we have timeseries data, such as: video frame processing ([paper](https://arxiv.org/pdf/1909.05622.pdf) & [code](https://github.com/matinhosseiny/Inception-inspired-LSTM-for-Video-frame-Prediction)), writing generator ([article w/ code](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/), [article w/ code](https://medium.com/towards-artificial-intelligence/sentence-prediction-using-word-level-lstm-text-generator-language-modeling-using-rnn-a80c4cda5b40)) as it can “keep in mind” something that happened many frames/ sentences ago. 

## 4.3.	Gated Recurrent Unit (GRU) Neural Networks - [Coursera](https://www.coursera.org/lecture/nlp-sequence-models/gated-recurrent-unit-gru-agZiL)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann9.png" width="250" height="180" > 

**GRU** is a subcategory of RNN. GRUs are similar with LSTMs, but with a different type of gates. The lack of output gate makes it easier to repeat the same output for a concrete input multiple time and, therefore, they are less resource consuming than LSTMs and have similar performance.

**Where do we use Gated Recurrent Unit?**

They are currently used in similar applications as LSTMs.

# 5.	Auto-Encoder Networks

## 5.1. Auto-Encoder (AE) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann10.png" width="145" height="200"> 

**Auto-encoder networks** can be trained without supervision – covered in the next chapter!
Their structure with a number of hidden cells smaller than the number of input cells (and number of output cells equals number of input cells). 

The fact that AE is trained such that the output is as close as possible to the input, forces AEs to generalize data and search for common patterns.

**Where do we use Auto-Encoders?**

Auto-encoders can only answer questions like: "How do we summarize the data?", so they are used for classification, clustering and feature compression in problems like face recognition and acquiring semantic meaning of words.  

## 5.2. Variational Auto-Encoder (VAE) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann11.png" width="150" height="200" > 

**VAE** is a category of AE. While the Auto-Encoder compresses features, Variational Auto-Encoders compresses the probability. 
This change makes a VAE answer questions like _"How strong is the connection between the two things?”_, _“Should we divide in two parts or are they completely independent?"_.
 
In neural net language, a VAE consists of an encoder, a decoder, and a loss function. 
In probability model terms, the variational autoencoder refers to approximate inference in a latent Gaussian model where the approximate posterior and model likelihood are parametrized by neural nets (the inference and generative networks).

•	Encoder: 
  -	in the neural net world, the encoder is a neural network that outputs a representation z of data x. 
  -	In probability model terms, the inference network parametrizes the approximate posterior of the latent variables z. The inference network outputs parameters to the distribution q(z∣x).
  
•	Decoder: 
  -	in deep learning, the decoder is a neural net that learns to reconstruct the data x given a representation z. 
  -	In terms of probability models, the likelihood of the data x given latent variables z is parametrized by a generative network. The generative network outputs parameters to the likelihood distribution p(x∣z).
  
•	Loss function: 
  -	in neural net language, we think of loss functions. Training means minimizing these loss functions. But in variational inference, we maximize the ELBO (which is not a loss function). This leads to awkwardness like calling optimizer.minimize(-elbo) as optimizers in neural net frameworks only support minimization.

**Where do we use Variational Auto-Encoder?**

They are powerful generative models with vast applications, including generating fake human faces, and purely music composition.

Other paper & code.

## 5.3.	Denoising Auto-Encoder (DAE) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann12.png" width="150" height="200" > 

**DAE** is a category of AE. Auto-encoders sometimes fail to find the most proper features but rather adapts to the input data (example of over-fitting). 

The Noise Reduction Auto-Encoder (DAE) adds some noise to the input unit - changing data by random bits, arbitrarily shifting bits in the input, and so on. 
By doing this, a forced noise reduction auto-encoder reconstructs the output from a somewhat noisy input, making it more generic, forcing the selection of more common features.

**Where do we use Denoising Auto-Encoders?**

They are important for feature selection and extraction and the main usage of this network is to recover a clean input from a corrupted version, such as image denoising (super resolution) for medical purposes.

## 5.4.	Sparse Auto-Encoder (SAE) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann13.png" width="150" height="200" > 

**Sparse Auto-Encoder** is another form of auto-encoding that sometimes pulls out some hidden aspects from the data. 

Here, the number of hidden cells is greater than the number of input or output cells and this constraint forces the model to respond to the unique statistical features of the input data.

**Where do we use Sparse Auto-Encoders?**

This type of auto-encoders can be used in popularity prediction (as this paper studied the prediction of Instagram posts popularity), and machine translation.

# 6.	Markov Chain (MC) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann14.png" width="180" height="200" > 

**Markov Chain** is an old chart concept. It is not a typical neural networks. 
Each of its endpoints is assigned with a certain probability. 

**Where do we use Markov Chain?**

In the past, it’s been used to construct a text structure like "dear" appears after "Hello" with a probability of 0.0053%.
They can be used as **probability-based categories** (like Bayesian filtering), for **clustering** and also as finite state machines.

# 7.	Hopfield Network (HN) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann15.png" width="180" height="200" > 

**Hopfield network** is initially trained to store a number of patterns or memories.
 
It is then able to recognize any of the learned patterns by exposure to only partial or even some corrupted information about that pattern 
_i.e._ it eventually settles down and returns the closest pattern or the best guess. 

Like the human brain memory, the Hopfield network provides similar pattern recognition. 
Each cell serves as input cell before training, as hidden cell during training and as output cell when used.

**Where do we use Hopfield Network?**

As HNs are able to discern the information even if corrupted, they can be used for denoising and restoring inputs. Given a half of learned picture or sequence, they will return a full object.

# 8.	Boltzmann Networks

## 8.1. Boltzmann Machine (BM) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann16.png" width="180" height="190" > 

**Boltzmann Machine**, also known as **Stochastic Hopfield Network**, is a network of symmetrically connected neurons. It is the first network topology that successfully preserves the simulated annealing approach.

It is named after the Boltzmann distribution (also known as Gibbs Distribution) which is an integral part of Statistical Mechanics. 
BM are non-deterministic (or stochastic) generative Deep Learning models, very similar to Hopfield Network, with only two types of nodes — hidden and input. 
There are no output nodes! 

In training, BM updates units one by one instead of in parallel. When the hidden unit updates its status, the input unit becomes the output unit. This is what gives them non-deterministic feature. 

They don’t have the typical 1 or 0 type output through which patterns are learned and optimized using Stochastic Gradient Descent. They learn patterns without that capability, and this is what makes them special.

**Where do we use Boltzmann Machine?**

Multi-layered Boltzmann machines can be used for so-called Deep Belief Networks.

## 8.2.	Restricted Boltzmann Machine (RBM) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann17.png" width="150" height="200" > 

The **Restricted Boltzmann Machines** are very similar to BMs in structure, but constrained RBMs are allowed to be trained back-propagating like FFs (the only difference is that the RBM will go through the input layer once before data is backpropagated). 

**Where do we use Restricted Boltzmann Machine?**

Restricted Boltzmann machine is an algorithm useful for dimensionality reduction, collaborative filtering, feature learning and topic modeling with practical application, for example, in speech recognition. 

# 9.	Deep Belief Network (DBN) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann18.png" width="280" height="190" > 

**Deep Belief Network** is actually a number of Boltzmann Machines surrounded by VAE together. They can be linked together (when one neural network is training another), and data can be generated using patterns learned. 

**Where do we use Deep Belief Network?**

Deep belief networks can be used for feature detection and extraction.

# 10.	Convolutional Networks

## 10.1. Deep Convolutional Network (DCN) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann19.png" width="280" height="190" > 

**Deep Convolutional Network** has convolutional units (or pools) and kernels, each for a different purpose. Convolution kernels are actually used to process input data, and pooling layers are used to reduce unnecessary features.

**Where do we use Deep Convolutional Network?**

They are usually used for image recognition, running on a small part of the image (~ 20x20 pixels). 

There is a small window sliding over along the image, analyzing pixel by pixel. The data then flows to the convolution layer, which forms a funnel (compression of the identified features). 

In terms of image recognition, the first layer identifies the gradient, the second layer identifies the line, and the third layer identifies the shape, and so on, up to the level of a particular object. DFF is usually attached to the end of the convolution layer for future data processing.

## 10.2. Deconvolutional Neural Network (DNN) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann20.png"  width="200" height="210" > 

The **Deconvolution Neural Network** is the inverted version of Deep Convolutional Network. 
DNN can generate the vector as: [dog: 0, lizard: 0, horse: 0, cat: 1] after capturing the cat's picture, while DCN can draw a cat after getting this vector. 

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann20-1.png)

**Where do we use Deconvolutional Network?**

You could tell the network “cat” and it will try to project it’s understanding of the features of a “cat”. DNN by itself is not entirely powerful, but when used in conjunction with some other structures, it can become very useful.

## 10.3. Deep Convolutional Inverse Graphics Network (DCIGN)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann21.png"> 

**Deep Convolutional Inverse Graphics Network** is a structure that connects a Convolutional Neural Network with a Deconvolutional Neural Network.
It might be confusing to call it a network when it is actually more of a Variational Auto-encoder (VAE).

**Where do we use Deep Convolutional Inverse Graphics Network?**

Most of these networks can be used in image processing and can process images that they have not been trained on before. 
They can be used to remove something from a picture, redraw it, or replace a horse with a zebra like the famous CycleGAN.
There are also many other types such as atrous convolutions and separable convolutions, that you can learn more about here.

# 11.	Generative Adversarial Neural Networks (GAN)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann22.png"> 

The **Generative Adversarial Network** represents a dual network consisting of generators and differentiator. Imagine two networks in competition, each trying to outsmart the other. The generator tries to generate some data, and the differentiator tries to discern which are the samples and which ones are generated (code). 

As long as you can maintain the balance between the training of the two neural networks, this architecture can generate the actual image.

**Where do we use Generative Adversarial Network?**

They are used in text to **image generation** ([paper](), [code]()), **image to image translation** ([paper](), [code]()), **increasing image resolution** ([paper](), [code]()) and **predicting next video frame** ([paper](), [code]()).

# 12.	Liquid State Machine (LSM) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann23.png"> 

**Liquid State Machines** are sparse neural networks whose activation functions are replaced (not all connected) by thresholds. When the threshold is reached, the cell accumulates the value information from the successive samples and the output freed, then again sets the internal copy to zero. 

**Where do we use Liquid State Machine?**

These neural networks are widely used in computer vision, speech recognition systems, but has no major breakthrough.

# 13.	Extreme Learning Machine (ELM) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann24.png"> 

**Extreme Learning Machines** reduce the complexity behind a feedforward network by creating a sparse, random connection of hidden layers. 

They require less computer power, and the actual efficiency depends very much on tasks and data.

**Where do we use Extreme Learning Machine?**

It is widely used in batch learning, sequential learning, and incremental learning because of its fast and efficient learning speed, fast convergence, good generalization ability, and ease of implementation.
However, due to its memory-residency, and high space and time complexity, the traditional ELM is not able to train big data fast and efficiently.

# 14.	Echo State Neural Networks (ESN)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann25.png"> 

**Echo status network** is a subdivision of a repeating network. Data passes through the input, and if multiple iterations are monitored, only the weight between hidden layers is updated after that.

**Where do we use Echo State Network?**

Besides multiple theoretical benchmarks, there is not any practical use of this Network. 

# 15.	Deep Residual Neural Networks (DRN) 

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann26.png"> 

**Deep Residual Network** (or **ResNet**) passes parts of input values to the next level. This feature makes it possible to reach many layers (up to 300), but they are actually recurrent neural network without a clear delay. 

**Where do we use Deep Residual Network?**

As the Microsoft Research study proves, Deep Residual Networks can be used with a significantly importance in image recognition.

# 16.	Kohonen Neural Networks (KN)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann27.png"> 

**Kohonen Network**, also known as Self-Organizing Map (SOM), introduces the "cell distance" feature. This network tries to adjust its cells to make the most probable response to a particular input. When a cell is updated, the closest cells are also updated.

They are not always considered "real" neural networks. 

**Where do we use Kohonen Network?**

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann27-1.png"> 

Kohonen Network produces a low-dimensional (typically two-dimensional) representation of the input space, called a map, and is therefore a method to do dimensionality reduction. 

A [practical example](https://en.wikipedia.org/wiki/Self-organizing_map#/media/File:Synapse_Self-Organizing_Map.png) of appliance is this map (by Original uploader Denoir at en.wikipedia) showing U.S. Congress voting patterns. 

The input data was a table with a row for each member of Congress, and columns for certain votes containing each member's yes/no/abstain vote. 

The SOM algorithm arranged these members in a two-dimensional grid placing similar members closer together. 

# 17.	Support Vector Machine (SVM) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann28.png"> 

**Support Vector Machines** are used for binary categorical work and the result will be "yes" or "no" regardless of how many dimensions or inputs the network processes.

It uses a technique called the kernel trick to do some extremely complex data transformations, then figures out how to separate input data based on the defined output labels. 

**Where do we use Support Vector Machine?**

SVMs should be the first choice for any classification task, because is one of the most robust and accurate algorithm among the other classification algorithms.

# 18.	Neural Turing Machine (NTM) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann29.png"> 

Neural networks are like black boxes - we can train them, get results, enhance them, but most of the actual decision paths are not visible to us.

The **Neurological Turing Machine (NTM)** is trying to solve this problem - it is an FF after extracting memory cells. Some authors also say that it is an abstract version of LSTM.

Memory is content based; this network can read memory based on the status quo, write memory, and also represents the Turing complete neural network.

**Where do we use Neural Turing Machine?**

A Neurological Turing Machine with a long short-term memory (LSTM) network controller can infer simple algorithms such as copying, sorting, and associative recall from examples alone.

Other open source implementations of NTMs exist but are not sufficiently stable for production use.


## What is Machine Learning?

**Machine Learning (ML)** is the computer programming field where the machine is led to learn from data. In contrast with traditional programming approaches where the developer had to continuously improve the code, machine learning aims to keep up with this ever-changing world by self-adapting and learning on the fly.  

It is been around for decades and nowadays it is present in so many shapes that became unnoticeable and yet indispensable for our daily life. From call center robots to a simple Google search, as well as Amazon or Netflix recommendations, they all have a machine learning algorithm working behind it. 

Every such application uses a specific learning system and we can categorize these systems by different criteria. We call them _supervised, unsupervised, semi-supervised, self-supervised_ or _reinforcement learning_ by the level of human supervision, _online_ or _batch learning_ depending on weather they are pretrained or learn on-the-spot and _instance-based_ or _model-based learning_ if it compares receiving data to known data points, or if otherwise detects patterns in the training data and builds a predictive model.

### Machine Learning Categories by the level of human supervision:
 
### 1. Supervised Learning

Supervised learning is the most common method because of its advantage of using known target to correct itself. 

Inspired by how students are supervised by their teacher who provides them the right answer to a problem, similarly this technique uses pre-matching input-output pairs. In this way, explicit examples make the learning process easy and straightforward. 

**When do we use Supervised Learning?** 

Whenever the problem lies in one of the two subcategories: **regression** or **classification**. 

**Regression** is the task of estimating or predicting continuous data (unstable values), such as: popularity ([paper](https://arxiv.org/pdf/1907.01985.pdf) & [code](https://github.com/dingkeyan93/Intrinsic-Image-Popularity))/ population growth/ weather ([article](https://stackabuse.com/using-machine-learning-to-predict-the-weather-part-1/) & [code](https://github.com/MichaelE919/machine-learning-predict-weather))/ stock prices ([code & details](https://github.com/dduemig/Stanford-Project-Predicting-stock-prices-using-a-LSTM-Network/blob/master/Final%20Project.ipynb)), etc. using algorithms like linear regression (because it outputs a probabilistic value, ex.: 40% chance of rain), non-linear regression or [Bayesian linear regression](https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7).

- If the model is unable to provide accurate results, backward propagation (detailed in the next chapter) is used to repeat the whole function until it receives satisfactory results.

**Classification** is the task of estimating or predicting discrete values (static values), processes which assigns meaning to items (tags, annotations, topics, etc.), having applications such as: image classification, spam detection, etc. using algorithms like _linear regression, Decision Tree_ and _Random Forests_. 

### Supervised Learning in different areas:

### Text (Natural Language Processing - NLP)

- **Machine Translation**: Syntactically Supervised Transformers for Faster Neural Machine Translation - [paper](https://arxiv.org/pdf/1906.02780v1.pdf) & [code](https://github.com/dojoteef/synst).
- **Named entity recognition**: Distantly Supervised Named Entity Recognition using Positive-Unlabeled Learning - [paper](https://arxiv.org/pdf/1906.01378v2.pdf) & [code](https://github.com/v-mipeng/LexiconNER).
- **Text Summarization**: Iterative Document Representation Learning Towards Summarization with Polishing - [paper](https://arxiv.org/pdf/1809.10324v2.pdf) & [code](https://github.com/yingtaomj/Iterative-Document-Representation-Learning-Towards-Summarization-with-Polishing).

### Image (Computer Vision)

- **Semantic Segmentation**: ResNeSt: Split-Attention Networks - [paper](https://arxiv.org/pdf/2004.08955v1.pdf) & code [Tensorflow](https://github.com/dmlc/gluon-cv)/[PyTorch](https://github.com/zhanghang1989/ResNeSt).
- **Image Classification**: Dynamic Routing Between Capsules - [paper](https://arxiv.org/pdf/1710.09829.pdf) & [code](https://github.com/Sarasra/models/tree/master/research/capsules).
- **Visual Question Answering**: Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning - [paper](https://arxiv.org/pdf/1703.06585v2.pdf) & [code](https://github.com/batra-mlp-lab/visdial-rl).
- **Person Re-identification**: Weakly supervised discriminative feature learning with state information for person identification - [paper](https://arxiv.org/pdf/2002.11939v1.pdf) & [code](https://github.com/KovenYu/state-information).

### Audio (Automatic Speech Recognition - ASR)

- **Speech to Text/ Text to Speech**: Speech to text and text to speech recognition systems-Areview - [paper](https://www.iosrjournals.org/iosr-jce/papers/Vol20-issue2/Version-1/E2002013643.pdf).
- **Speech recognition**: Deep Speech 2: End-to-End Speech Recognition in English and Mandarin - [paper](https://arxiv.org/pdf/1512.02595v1.pdf) & [code](https://github.com/tensorflow/models/tree/master/research/deep_speech), Real-Time Voice Cloning - [code](https://github.com/CorentinJ/Real-Time-Voice-Cloning).
- **Speech Synthesis**: Natural TTS Synthesis by conditioning wavenet on MEL spectogram predictions - [paper](https://arxiv.org/pdf/1712.05884v2.pdf) & [code](https://github.com/NVIDIA/tacotron2) & [explained](https://github.com/codetendolkar/tacotron-2-explained)(using Tacotron 2 method); Other method: [WaveNet](https://github.com/r9y9/wavenet_vocoder) - [paper](https://arxiv.org/pdf/1609.03499v2.pdf) & [code](https://github.com/maciejkula/spotlight).
- **Speeche Enhancement**: Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression - [paper](https://arxiv.org/pdf/2005.07551.pdf) & [code](https://github.com/breizhn/DTLN).
- **Speaker Verification**: Text Independant Speaker Verification - [code](https://github.com/Suhee05/Text-Independent-Speaker-Verification).

### 2.	Unsupervised Learning

Unsupervised learning, on the other hand, is dealing with unlabeled datasets, being forced to find patterns on its own by extracting useful features from provided data and analyzing its structure.

**When do we use Unsupervised Learning?**

Unsupervised learning is applied when the dataset doesn’t come with labels, as well as when the labels are available, but you seek for more interesting hidden patterns in your data. 

This learning method is being used for tasks such as: _clustering, data visualization, dimentionality reduction_ and _anomaly detection._

**Clustering**: is the task for identifying similarity among items in order to group them – without having a name for that group (a label).
   Popular algorithms for this task: **K-Mean, KNN, DBSCAN, Hierarchical Cluster Analysis (HCA)**

**Visualization**: is the task for identifying and providing qualitative understanding of your dataset, like: trends, outliers, and patterns.
   Popular algorithms for this task: **Principal Component Analysis (PCA), Kernel PCA, Locally Linear Embedding, t-Distributed Stochastic Neighbor Embedding**; They conserve as much structure as they can by keeping separate classes in the input to prevent overlapping in the visualization, so that you can identify unusual patterns, if present. 
 
**Dimensionality reduction** (essential in meaningful compression and structure discovery) has the goal to simplify the input data without losing too much information. A solution is to merge several similar features into one. For example, a movie’s director may be strongly correlated with its actors, so the dimensionality reduction algorithm will merge them into one feature that represents the movie staff. This is called _feature extraction_. 

   - a dimensionality reduction algorithm used before a learning method will allow a much faster running, occupy less memory space and, sometimes, might perform better. 

**Anomaly Detection** has the goal to detect any unusual activity or presence in your data. Such algorithms detect credit card frauds, sustain the system health monitoring, etc. Even if you don’t have such a complex application, you can still run an anomaly detection algorithm to make sure the training set is not misleading. 

   - An anomaly detection algorithm used before a learning method will eliminate possible outliers, improving the dataset quality.

### Unsupervised Learning in different areas:

### Text (Natural Language Processing - NLP)

- **Language Modelling**: Improving Language Understanding by Generative Pre-Training - [paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) & [code](https://github.com/openai/finetune-transformer-lm).
- **Machine Translation**: Unsupervised Neural Machine Translation with Weight Sharing - [paper](https://arxiv.org/pdf/1804.09057.pdf) & [code](https://github.com/facebookresearch/UnsupervisedMT).
- **Text Classification**: Unsupervised Text Classification for Natural Language Interactive Narratives - [paper](https://people.ict.usc.edu/~gordon/publications/INT17A.PDF) & code not provided.
- **Question Answering**: Language Models are Unsupervised Multitask Learner - [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) & [code](https://github.com/hanchuanchuan/gpt-2).
- **Abstractive Summarization**: Centroid-based Text Summarization through Compositionality of Word Embeddings - [paper](https://www.aclweb.org/anthology/W17-1003.pdf) & [code](https://github.com/gaetangate/text-summarizer).

### Image (Computer Vision)

- **Semantic Segmentation**: Invariant Information Clustering for Unsupervised Image Classification and Segmentation - [paper](https://arxiv.org/pdf/1807.06653.pdf) & [code](https://github.com/xu-ji/IIC).
- **Image Classification**: SCAN: Learning to Classify Images without Labels - [paper](https://arxiv.org/pdf/2005.12320v2.pdf) & [code](https://github.com/wvangansbeke/Unsupervised-Classification).
- **Object Recognition**: Unsupervised Domain Adaptation through Inter-modal Rotation for RGB-D Object Recognition - [paper](https://arxiv.org/pdf/2004.10016v1.pdf) & [code](https://github.com/MRLoghmani/relative-rotation).
- **Person Re-identification**: Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification - [paper](https://arxiv.org/pdf/1811.10144v3.pdf) & [code](https://github.com/SHI-Labs/Self-Similarity-Grouping).

### Audio (Automatic Speech Recognition - ASR)

- **Speech to Text/ Text to Speech**: Representation Learning with Contrastive Predictive Coding - [paper](https://arxiv.org/pdf/1807.03748v2.pdf) & [code](https://github.com/davidtellez/contrastive-predictive-coding).
- **Speech recognition**: A segmental framework for fully-unsupervised large-vocabulary speech recognition - [paper](https://arxiv.org/pdf/1606.06950v2.pdf) & [code](https://github.com/kamperh/recipe_bucktsong_awe).
- **Speech Synthesis**: Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis - [paper](https://arxiv.org/pdf/1803.09017v1.pdf) & [code](https://github.com/syang1993/gst-tacotron).
- **Speeche Enhancement**: Supervised and Unsupervised Speech Enhancement
Using Nonnegative Matrix Factorization - [paper](https://arxiv.org/pdf/1709.05362v1.pdf) & [code](https://github.com/mohammadiha/bnmf).
- **Speaker Verification**: An Unsupervised Autoregressive Model for Speech Representation Learning - [paper](https://arxiv.org/pdf/1904.03240v2.pdf) & [code](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding).

### 3.	Semi-supervised Learning

Semi-supervised learning is the method used when the training dataset has both labeled and unlabeled data.

An example from everyday life where we meet this kind of machine learning is in the photo-storage cloud-based services. You might have noticed that once you upload your photos to a cloud service, it automatically makes a distinction between different people in your pictures. Furthermore, it asks you to add a tag for each person (which will represent the labeled data) so that it can learn to name them in other untagged pictures uploaded (which is the unlabeled data).

**When do we use Semi-Supervised Learning?**

Semi-Supervised learning is used when have both labeled and unlabeled data.

Some Semi-Supervised Algorithms include: **self-training, generative methods, mixture models, graph-based methods, co-training, semi-supervised SVM** and many others. 

### Semi-supervised Learning in different areas:

### Text (Natural Language Processing - NLP)

- **Language Modelling**: Semi-supervised sequence tagging with bidirectional language models - [paper](https://arxiv.org/pdf/1705.00108v1.pdf) & [code](https://github.com/flairNLP/flair).
- **Machine Translation**: A Simple Baseline to Semi-Supervised Domain Adaptation for Machine Translation - [paper](https://arxiv.org/pdf/2001.08140v2.pdf) & [code](https://github.com/jind11/DAMT).
- **Text Classification**: Variational Pretraining for Semi-supervised Text Classification - [paper](https://arxiv.org/pdf/1906.02242v1.pdf) & [code](https://github.com/allenai/vampire).
- **Question Answering**: Addressing Semantic Drift in Question Generation for Semi-Supervised Question Answering - [paper](https://arxiv.org/pdf/1909.06356v1.pdf) & [code](https://github.com/ZhangShiyue/QGforQA).
- **Abstractive Summarization**: Abstractive and Extractive Text Summarization using Document Context Vector and Recurrent Neural Networks - [paper](https://arxiv.org/pdf/1807.08000.pdf) & code not provided.

### Image (Computer Vision)

- **Semantic Segmentation**: Semi-supervised semantic segmentation needs strong, varied perturbations - [paper](https://arxiv.org/pdf/1906.01916v4.pdf) & [code](https://github.com/Britefury/cutmix-semisup-seg).
- **Image Classification**: Fixing the train-test resolution discrepancy - [paper](https://arxiv.org/pdf/2003.08237v4.pdf) & [code](https://github.com/facebookresearch/FixRes).
- **Object Recognition**: Data Distillation: Towards Omni-Supervised Learning - [paper](https://arxiv.org/pdf/1712.04440v1.pdf) & [code](https://github.com/facebookresearch/detectron) & [code](Data Distillation: Towards Omni-Supervised Learning).
- **Person Re-identification**: Sparse Label Smoothing Regularization for Person Re-Identification - [paper](https://arxiv.org/pdf/1809.04976v3.pdf) & [code](https://github.com/jpainam/SLS_ReID).

### Audio (Automatic Speech Recognition - ASR)

- **Speech to Text/ Text to Speech**: Libri-Light: A Benchmark for ASR with Limited or No Supervision - [paper]() & [code](https://github.com/facebookresearch/libri-light).
- **Speech recognition**: Semi-Supervised Speech Recognition via Local Prior Matching - [paper](https://arxiv.org/pdf/2002.10336v1.pdf) & [code](https://github.com/facebookresearch/wav2letter).
- **Speech Synthesis**: Semi-Supervised Generative Modeling for Controllable Speech Synthesis - [paper](https://arxiv.org/pdf/1910.01709v1.pdf) & code not provided.
- **Speeche Enhancement**: Semi-Supervised Multichannel Speech Enhancement With a Deep Speech Prior - [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8861142) & [code](https://github.com/sekiguchi92/SpeechEnhancement).
- **Speaker Verification**: Learning Speaker Representations with Mutual Information - [paper](https://arxiv.org/pdf/1812.00271v2.pdf) & [code](https://github.com/Js-Mim/rl_singing_voice).


### 4.	Self-supervised Learning

Self-supervised learning is the method with a level of supervision similar with the fully supervised method, but the labels here are automatically generated, typically by a heuristic algorithm, from the input data. After the label extraction, the following steps are similar as in the supervised learning algorithm.

Supervised learning is a safe bet, but it is limited. Unsupervised and Self-supervised learning are more flexible options and bring considerable value to the data.

**When do we use Self-Supervised Learning?**

Self-Supervised Learning is mostly use for motion-object detection as in this [paper](https://arxiv.org/pdf/1905.11137.pdf) & [code](https://people.eecs.berkeley.edu/~pathak/unsupervised_video/). Here is [a list of other papers](https://github.com/jason718/awesome-self-supervised-learning) using self-supervised learning.

### Self-supervised Learning in different areas:

### Text (Natural Language Processing - NLP)

- **Language Modelling**: ALBERT: A Lite BERT for Self-supervised Learning of Language Representations - [paper](https://arxiv.org/pdf/1909.11942v6.pdf) & [code](https://github.com/tensorflow/models/tree/master/official/nlp/albert).
- **Machine Translation**: Self-Supervised Neural Machine Translation - [paper](https://www.aclweb.org/anthology/P19-1178.pdf) & code not provided.
- **Text Classification**: Supervised Multimodal Bitransformers for Classifying Images and Text - [paper](https://arxiv.org/pdf/1909.02950v1.pdf) & [code](https://github.com/huggingface/transformers).
- **Abstractive Summarization**: PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization - [paper](https://arxiv.org/pdf/1912.08777v2.pdf) & [code](https://github.com/google-research/pegasus).

### Image (Computer Vision)

- **Semantic Segmentation**: Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation - [paper](https://arxiv.org/pdf/2004.04581v1.pdf) & [code](https://github.com/YudeWang/SEAM).
- **Image Classification**: Self-Supervised Learning For Few-Shot Image Classification - [paper](https://arxiv.org/pdf/1911.06045v2.pdf) & [code](https://github.com/phecy/SSL-FEW-SHOT).
- **Action Recognition**: A Multigrid Method for Efficiently Training Video Models - [paper](https://arxiv.org/pdf/1912.00998v2.pdf) & [code](https://github.com/facebookresearch/SlowFast).
- **Person Re-identification**: Enhancing Person Re-identification in a Self-trained
Subspace - [paper](https://arxiv.org/pdf/1704.06020v2.pdf) & [code](https://github.com/Xun-Yang/ReID_slef-training_TOMM2017).

### Audio (Automatic Speech Recognition - ASR)

- **Speech recognition**: Multi-task self-supervised learning for Robust Speech Recognition - [paper](https://arxiv.org/pdf/2001.09239v2.pdf) & [code](https://github.com/santi-pdp/pase).
- **Speeche Enhancement**: More Grounded Image Captioning by Distilling Image-Text Matching Model - [paper](https://arxiv.org/pdf/2004.00390v1.pdf) & [code](https://github.com/YuanEZhou/Grounded-Image-Captioning).
- **Speaker Verification**: AutoSpeech: Neural Architecture Search for Speaker Recognition - [paper](https://arxiv.org/pdf/2005.03215v1.pdf) & [code](https://github.com/TAMU-VITA/AutoSpeech).


### 5.	Reinforcement Learning (RL) 

Reinforcement learning has no kind of human supervision. It is a completely different approach, where the machine – called the agent – learns by observing the environment: it selects actions from a list of possibilities and acts accordingly, getting a reward if the result is good or a penalty if the result is bad. 
 
**When do we use Reinforcement Learning?**

Reinforcement learning is used in Games ([DeepMind’s AlphaGo](https://deepmind.com/research/case-studies/alphago-the-story-so-far) – from Google: [paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ), the official code hasn’t been released, but here’s an [alternative](https://github.com/tensorflow/minigo)), Real-Time Decisions (Traffic Light Control – [paper](http://web.eecs.utk.edu/~ielhanan/Papers/IET_ITS_2010.pdf), [paper](https://arxiv.org/pdf/1903.04527.pdf) & [code](https://github.com/cts198859/deeprl_network/blob/master/README.md)), Robot Navigation ([MuJoCo](http://www.mujoco.org/book/index.html) – physics simulator), Skill Acquisition (Self-Driving Car – [paper](https://arxiv.org/pdf/1801.02805.pdf) & [code](https://github.com/lexfridman/deeptraffic)). 
