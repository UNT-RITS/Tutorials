# How to code Neural Network models in Python

## :construction: ... Work in Progress ... :construction:

Content:

- [What is an Artificial Neural Network?]()
    - [What are the main components and why do we need each of them?]()
        - Weights, Bias and Layers
        - Activation Function: Linear Activation Function and Non-linear Activation Function (Sigmoid, Tanh & ReLU)
        - Derivatives
    - [Architectures of Artificial Neural Network]()
    
- [What is Machine Learning?]()

## What is an Artificial Neural Network?
**Artificial Neural networks (ANN)** are a set of algorithms, modeled in a similar way the human brain works, developed to recognize and predict patterns. They interpret given data through a machine perception, using labeling or collecting raw input. The patterns they recognize are numerical, expressed as vectors, and so is the output before having assigned a meaning (check this [video](https://www.youtube.com/watch?v=aircAruvnKk) explanation). 
Therefore, it is essential to convert the real-world input data, like images, sounds or text, into numerical values.

Most of the existing neural networks architectures are shown in the following picture:
![ANNs](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/ann1.png)

Deep learning is the name we use for _“stratified neural networks”_ = **networks composed of several layers**.

The layers are represented by a column vector and the elements of the vector can be thought of as _nodes_ (also named _neurons_ or _units_).

A **node (neuron)** is just **a place where computation happens**. It receives input from some other nodes, or from an external source and computes the outcome. Each input has an associated **weight (w)**, which is assigned on the basis of its relative importance to other inputs. Then, the output is computed by combining a set of **coefficients**, or **weights**, that either _amplify_ or _reduce_ that input depending on its importance. 

The input-weights are multiplied between each other and summed up. The sum is passed through a node’s so-called **activation function**, to determine whether or to what extent that signal should progress further through the network to affect the ultimate outcome. If the signal passes through, the neuron has been **activated**.

### What are the main components and why do we need each of them?

#### 1. Why do we need Weights, Bias and Layers?

**Weight** shows the strength of the particular node. In other words, the weight is the assigned significance of an input in comparison with the relative importance of other inputs.

A **bias** value allows you to shift the activation function curve up or down.

A neural network can usually consist of three types of nodes:

   - **Input Nodes** – they provide information from the outside world to the network and are referred to as the “Input Layer”. No computation is performed in any of the Input nodes – they just pass on the information to the hidden nodes.

   - **Hidden Nodes** – they have no direct connection with the outside world and form a so called “Hidden Layer”. They perform computations and transfer information from the input nodes to the output nodes.

   - **Output Nodes** – they are collectively referred to as the “Output Layer” and are responsible for computations and mapping information from the network to the outside world.

#### 2. Why do we need Activation Function?

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

#### 3. Why the derivative/differentiation is being used?

We use differentiation in almost every part of Machine Learning and Deep Learning, because when updating the curve, we need to know in which direction and how much to change or update the curve depending upon the slope. 

In the following table it is a clear distinction and classification of some functions and their derivates.

![ANNs](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/ann2.png)

## Architectures of Artificial Neural Network


## What is Machine Learning?

Machine Learning (ML) is the computer programming field where the machine is led to learn from data. In contrast with traditional programming approaches where the developer had to continuously improve the code, machine learning aims to keep up with this ever-changing world by self-adapting and learning on the fly.  

It is been around for decades and nowadays it is present in so many shapes that became unnoticeable and yet indispensable for our daily life. From call center robots to a simple Google search, as well as Amazon or Netflix recommendations, they all have a machine learning algorithm working behind it. 

Every such application uses a specific learning system and we can categorize these systems by different criteria. We call them _supervised, unsupervised, semi-supervised, self-supervised_ or _reinforcement learning_ by the level of human supervision, _online_ or _batch learning_ depending on weather they are pretrained or learn on-the-spot and _instance-based_ or _model-based learning_ if it compares receiving data to known data points, or if otherwise detects patterns in the training data and builds a predictive model.

### 1.	Supervised Learning

Supervised learning is the most common method because of its advantage of using known target to correct itself. 

Inspired by how students are supervised by their teacher who provides them the right answer to a problem, similarly this technique uses pre-matching input-output pairs. In this way, explicit examples make the learning process easy and straightforward. 

**When do we use Supervised Learning?** 

Whenever the problem lies in one of the two subcategories: **regression** or **classification**. 

**Regression** is the task of estimating or predicting continuous data (unstable values), such as: popularity (paper & code)/ population growth/ weather (article & code)/ stock prices (code & details), etc. using algorithms like linear regression (because it outputs a probabilistic value, ex.: 40% chance of rain), non-linear regression or Bayesian linear regression.

- If the model is unable to provide accurate results, backward propagation (detailed in the next chapter) is used to repeat the whole function until it receives satisfactory results.

**Classification** is the task of estimating or predicting discrete values (static values), processes which assigns meaning to items (tags, annotations, topics, etc.), having applications such as: image classification, spam detection, etc. using algorithms like _linear regression, Decision Tree_ and _Random Forests_. 

### 2.	Unsupervised Learning

Unsupervised learning, on the other hand, is dealing with unlabeled datasets, being forced to find patterns on its own by extracting useful features from provided data and analyzing its structure.

**When do we use Unsupervised Learning?**

Unsupervised learning is applied when the dataset doesn’t come with labels, as well as when the labels are available, but you seek for more interesting hidden patterns in your data. 

This learning method is being used for tasks such as: _clustering, data visualization, dimentionality reduction_ and _anomaly detection._

**Clustering**: is the task for identifying similarity among items in order to group them – without having a name for that group (a label).
   Popular algorithms for this task: K-Mean, KNN, DBSCAN, Hierarchical Cluster Analysis (HCA);

**Visualization**: is the task for identifying and providing qualitative understanding of your dataset, like: trends, outliers, and patterns.
   Popular algorithms for this task: Principal Component Analysis (PCA), Kernel PCA, Locally Linear Embedding, t-Distributed Stochastic Neighbor Embedding; They conserve as much structure as they can by keeping separate classes in the input to prevent overlapping in the visualization, so that you can identify unusual patterns, if present. 
 
**Dimensionality reduction** (essential in meaningful compression and structure discovery) has the goal to simplify the input data without losing too much information. A solution is to merge several similar features into one. For example, a movie’s director may be strongly correlated with its actors, so the dimensionality reduction algorithm will merge them into one feature that represents the movie staff. This is called _feature extraction_. 

   - a dimensionality reduction algorithm used before a learning method will allow a much faster running, occupy less memory space and, sometimes, might perform better. 

**Anomaly Detection** has the goal to detect any unusual activity or presence in your data. Such algorithms detect credit card frauds, sustain the system health monitoring, etc. Even if you don’t have such a complex application, you can still run an anomaly detection algorithm to make sure the training set is not misleading. 

   - An anomaly detection algorithm used before a learning method will eliminate possible outliers, improving the dataset quality.

Some specific areas include recommender systems, targeted marketing and customer segmentation, big data visualization, etc.

### 3.	Semi-supervised Learning

Semi-supervised learning is the method used when the training dataset has both labeled and unlabeled data.

An example from everyday life where we meet this kind of machine learning is in the photo-storage cloud-based services. You might have noticed that once you upload your photos to a cloud service, it automatically makes a distinction between different people in your pictures. Furthermore, it asks you to add a tag for each person (which will represent the labeled data) so that it can learn to name them in other untagged pictures uploaded (which is the unlabeled data).

**When do we use Semi-Supervised Learning?**

Semi-Supervised learning is used when have both labeled and unlabeled data.

Some Semi-Supervised Algorithms include:

▪ Self-Training 

▪ Generative methods, mixture models 

▪ Graph-based methods

▪ Co-Training 

▪ Semi-supervised SVM 

▪ Many others 

### 4.	Self-supervised Learning

Self-supervised learning is the method with a level of supervision similar with the fully supervised method, but the labels here are automatically generated, typically by a heuristic algorithm, from the input data. After the label extraction, the following steps are similar as in the supervised learning algorithm.

Supervised learning is a safe bet, but it is limited. Unsupervised and Self-supervised learning are more flexible options and bring considerable value to the data.

**When do we use Self-Supervised Learning?**

Self-Supervised Learning is mostly use for motion-object detection as in this paper & code. Here is a list of other papers using self-supervised learning.


### 5.	Reinforcement Learning (RL) 

Reinforcement learning has no kind of human supervision. It is a completely different approach, where the machine – called the agent – learns by observing the environment: it selects actions from a list of possibilities and acts accordingly, getting a reward if the result is good or a penalty if the result is bad. 
 
**When do we use Reinforcement Learning?**

Reinforcement learning is used in Games (DeepMind’s AlphaGo – from Google: paper, the official code hasn’t been released, but here’s an alternative), Real-Time Decisions (Traffic Light Control – paper, paper & code), Robot Navigation (MuJoCo – physics simulator), Skill Acquisition (Self-Driving Car – paper & code). 
