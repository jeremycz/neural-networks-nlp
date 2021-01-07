# Neural Networks for NLP

Notes and code related to neural networks for NLP as part of CS224N
## Useful Links

- [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/)
- [Sebastian Ruder](https://ruder.io/)

## Softmax classifier

- Training data: $\{x_i, y_i\}^N_{i=1}$
- Approach: Assume $x_i$ are fixed, train softmax/logistic regression weights $W\in \mathcal{R}^{C\times d}$ to determine a decision boundary (hyperplane)
  - Note slight differences between logistic and softmax regression
- Method: For each x, predict:

$$
P(y|x) = \frac{\exp(W_yx)}{\sum_{c=1}^C\exp(W_cx)}
$$

Break prediction function into two steps:

1. Take the $y$th row of $W$ and multiply that row with $x$:

$$
W_yx=\sum_{i=1}^dW_{yi}x_i=f_y
$$

Compute all $f_c$ for $c=1,\dots,C$

2. Apply softmax function to get normalised probability

$$
P(y|x) = \frac{\exp(f_y)}{\sum_{c=1}^C\exp(f_c)} = \textrm{softmax}(f_y)
$$

- Objective: Maximise probability of the correct class for each training example. Alternatively, minimise the negative log probability

$$
-\log P(y|x) = -\log\left(\frac{\exp(f_y)}{\sum_{c=1}^C\exp(f_c)}\right)
$$

### Cross-entropy relation to softmax regression

- Given a true probability distribution $p$ and a computed distribution $q$, the cross-entropy is

$$
H(p,q) = -\sum_{c=1}^Cp(c)\log q(c)
$$

- Assume a ground truth probability distribution that is 1 at the right class and 0 everywhere else $p = [0,\dots,0,1,0,\dots,0]$. As $p$ is one-hot, the only term left is the negative log probabiltiy of the true class.

### Classification over a full dataset

$$
\begin{aligned}
J(\theta) &= \frac{1}{N}\sum_{i=1}^N-\log\left(\frac{\exp(f_y)}{\sum_{c=1}^C\exp(f_c)}\right) \\
f &= Wx
\end{aligned}
$$

- Softmax/logistic regression alone not very powerful
  - Linear decision boundaries

## Classification with word vectors

Commonly in NLP deep learning:

- Learn both weights, $W$, and word vectors, $x$
- Learn both conventional parameters and representations

## Artificial neurons

$$
h_{w,b}(x) = f(w^Tx + b)
$$

where $f$ is an activation function.

Using matrix notation

$$
\begin{aligned}
z &= Wx + b
a &= f(z)
\end{aligned}
$$

- Activation is applied element-wise
- Non-linear activation functions used to approximate more complex functions

## NER

- Use context to resolve ambiguities (e.g. Paris - Paris, France vs. Paris Hilton)

Example: Binary classification with unnormalized scores - Collobert and Weston (2008, 2011)

- Classify whether the center word in a window is a location
  - $x_\textrm{window} = \begin{bmatrix} x_\textrm{museums} & x_\textrm{in} & x_\textrm{Paris} & x_\textrm{are} & x_\textrm{amazing} \end{bmatrix}$
  - If center element of window is a location, then 'true', otherwise 'false'

$$
\textrm{score}(x) = s = U^Ta\in \mathcal{R}
$$

- Convert vector of activations into a scalar score
- Use a 3-layer neural net

$$
\begin{aligned}
s &= U^Tf(Wx+b) \\
&= x\in \mathcal{R}^{wd\times 1}, W\in\mathcal{R}^{8\times wd}, U\in\mathcal{R}^{8\times 1}
\end{aligned}
$$

- $w$ - window size
- $d$ - size of embedding vector
- $x$ - concatenated embeddings of tokens in window
- 8 neurons in intermediate layer

### Window classification

- Idea: Classify a word in its context window of neighbouring words
- A simple way to classify a word in context might be to average the word vectors in the window and to classify the average vector
  - Problem: Lose position information
- Alternative: Concatenate vectors in window into a column vector $x \in \mathcal{R}^{5d}$

### Maximum Margin Objective Function

- Ensure score computed for 'true' data points is higher than that for 'false' labelled data points
- The max-margin objective function is most commonly associated with SVMs

Example:

- True window: "Museums in Paris are amazing" - score $s$
- False/corrupted window: "Not all museums in Paris" - score $s_c$

Objective: Maximise $(s-s_c)$, or minimise $(s_c-s)$

- Only compute error if $s_c > s -> (s_c-s) > 0$
  - We only care that the 'true' data point has a higher score than the 'false' data point

Updated objective:

$$
\min J = \max(s_c-s, 0)
$$

We can also add a margin of safety to ensure that the scores for 'true' datapoints are higher than those for 'false' datapoints by some margin $\Delta$.

$$
\min J = \max(\Delta+s_c-s, 0)
$$

We can scale this margin such that $\Delta = 1$ and let the parameters adapt to this without any change in performance.

$$
\min J = \max(1+s_c-s, 0)
$$

## Neural Networks: Tips and Tricks

- [Link to document](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes03-neuralnets.pdf)

### Gradient Check

- Use finite difference to check gradients (central preferred vs. forward difference)

### Regularization

- Penalises weights for being too large, reduces overfitting
- Can be interpreted as the prior Bayesian belief that the optimal weights are close to zero
- Do not regularise bias terms

$$
J_R = J + \lambda\sum_{i=1}^L\|W^{(i)}\|_F
$$

- $L_2$ regularization
- $|W^{(i)}\|_F$ is the Frobenius norm

$$
|U\|_F = \sqrt{\sum_i\sum_jU_{ij}^2}
$$

- Too high $\lambda$ - underfitting - model does not learn anything
- Too low $\lambda$ - effect of regularisation is diminished

Other regularisation types

- $L_1$ - sums over absolute values of the parameter elements
  - Less commonly applied - leads to sparsity of parameter weights

### Dropout

[Srivastava et al. (2014) Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR 15 1929-1958](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

- Another technique for regularisation
- During training, randomly drop with some probability $(1-p)$ a subset of neurons during each forward/backward pass
- Use full network to compute predictions
- Intuitive explanation - training exponentially many smaller networks at once and averaging over their predictions
  
Application

- Forward pass: Take the output $h$ of each layer of neurons, keep each neuron with probability $p$, else set it to $0$
- Backward pass: Only pass gradients through neurons that were kept alive during the forward pass

*In order for dropout to work effectively, the expected output of a neuron during testing should be approximately the same as it is during training - otherwise the magnitude of the outputs could be radically different, and the behaviour of the network is no longer well-defined. Thus, we must typically divide the outputs of each neuron during testing by a certain value.*

- Weights of the network will be larger than usual because of dropout. Thus, before testing, the weights must be scaled by the chosen dropout rate.
- The weights can be rescaled at training time instead, after each weight update at the end of a mini-batch.
  - Sometimes referred to as *inverse dropout*

### Activation functions

Sigmoid

- $\sigma(z)\in(0,1)$

$$
\begin{aligned}
  \sigma(z) &= \frac{1}{1+\exp(-z)} \\
  \sigma'(z) &= \sigma(z)(1-\sigma(z))
\end{aligned}
$$

Tanh

- Often found to converge faster in practice
- $\tanh(z)\in(-1,1)$

$$
\begin{aligned}
  \tanh(z) &= \frac{\exp(z)-\exp(-z)}{\exp(z)+\exp(-z)} = 2\sigma(2z) - 1\\
  \tanh'(z) &= 1-\tanh^2(z)
\end{aligned}
$$

Hard tanh

- Computationally cheaper
- Saturates for magnitudes of $z > 1$

$$
\begin{aligned}
  \textrm{hardtanh}(z) &= \begin{cases}
    -1 & z < -1 \\
    z & -1 \leq z \leq 1 \\
    1 & z > 1
  \end{cases} \\
  \textrm{hardtanh}'(z) &= \begin{cases}
    1 & -1 \leq z \leq 1 \\
    0 & \textrm{otherwise}
  \end{cases}
\end{aligned}
$$

Soft sign

- Alternative to tahn, does not saturate

$$
\begin{aligned}
  \textrm{softsign}(z) &= \frac{z}{1 + |z|} \\
  \textrm{softsign}'(z) &= \frac{\textrm{sgn}(z)}{(1+z)^2}
\end{aligned}
$$

where $\textrm{sgn}$ is the signum function which returns $\pm 1$ depending on the sign of $z$.

ReLU (Rectified Linear Unit)

- Does not saturate for larger values of $z$

$$
\begin{aligned}
  \textrm{relu}(z) &= \max(z, 0) \\
  \textrm{relu}'(z) &= \begin{cases}
    1 & z > 0 \\
    0 & \textrm{otherwise}
  \end{cases}
\end{aligned}
$$

Leaky ReLU

- Allows error propagation backwards when $z < 0$ compared to ReLU

$$
\begin{aligned}
  \textrm{leaky}(z) &= \max(z, kz) \\
  \textrm{leaky}'(z) &= \begin{cases}
    1 & z > 0 \\
    k & \textrm{otherwise}
  \end{cases}
\end{aligned}
$$

### Data Preprocessing

- Mean subtraction
  - Zero-center data
  - Mean calculated using training set only, and this value is subtracted from the training, validation and testing sets
- Normalisation
  - Scale every input feature dimension to have similar ranges of magnitudes
  - Divide features by standard deviation calculated using training set
- Whitening
  - Convert data to have identity covariance matrix i.e. features become uncorrelated and have a unit variance
  - Process:
    - Mean-subtract data, to get $X'$
    - Compute SVD of $X'$ to get $U$, $S$, $V$
    - Compute $UX'$ to project $X'$ into the basis defined by the columns of $U$
    - Divide each dimension of the result by the corresponding singular value in $S$ to scale data appropriately (if a singular value is zero, use some small number instead)

### Parameter Initialisation

[Xavier, Bengio (2010) Understanding the difficulty of training deep feedforward neural networks.](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi])

- Good starting strategy is to initialise the weights to small random numberes normally distributed around zero
- Xavier initialisation:
  - Empirical findings - faster convergence and lower error rates when using sigmoid/tanh activations with the following initialisation strategy

$$
W \backsim U\left[-\sqrt{\frac{6}{n^{(l)}+n^{(l+1)}}},\sqrt{\frac{6}{n^{(l)}+n^{(l+1)}}}\right]
$$

where $n^{(l)}$ is the number of input units to $W$ and $n^{(l+1)}$ is the number of output units from $W$. 

  - Biases are initialised to zero. 
  - This approach aims to maintain activation variances and backpropagated gradient variances across layers (gradient variances generally decrease with backpropagation across layers)

### Learning Strategies

Naive gradient descent:

$$
\theta^\textrm{new} = \theta^\textrm{old} - \alpha\nabla_\theta J(\theta)
$$

- High learning rate - overshoot, faster convergence not guaranteed
- Small learning rate - slow convergence, may get stuck in local minima
- Collobert - scale learning rate of a weight $W_{ij}$ by inverse square root of the fan-in (number of layer inputs) of the neuron, $n^{(l)}$

$$
W\in\mathcal{R}^{n^{(l+1)}\times n^{(l)}}
$$

#### Annealing

- Reduce learning rate after a set number of iterations
- Common methods: 
  - Reduce learning rate by a factor of $x$ after every $n$ iterations
  - Exponential decay - $\alpha(t) = \alpha_0\exp(-kt)$
  - Constant then decay - $\alpha(t) = \alpha_0\tau / \max(t, \tau)$

#### Momentum

$$
\begin{aligned}
  v &= \mu v - \alpha\nabla_\theta J(\theta) \\
  \theta &\leftarrow \theta + v
\end{aligned}
$$

- The gradient increments the previous velocity
- The velocity also decays by $\mu$ ($\mu < 1$, typically slightly less than 1)

#### Adaptive Optimization

- Learning rate can vary for each parameter
- Learning rate depends on the history of gradient updates - parameters with a scarce history of updates are updated faster using a larger learning rate

$$
\theta_{t,i} = \theta_{t-1,i} - \frac{\alpha}{\sqrt{\sum_{\tau=1}^t g_{\tau,i}^2} + \epsilon}g_{t,i}
$$

where

$$
g_{t,i} = \frac{\partial}{\partial\theta_t^i}J(\theta)
$$

- Learning rates are scaled by the RMS of the gradient history (low RMS - high learning rate)

- [AdaGrad - Duchi et al. (2011) Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. JMLR 12 2121-2159](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
  - As shown above
- [RMSPprop (unpublished work)](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
  - Version of AdaGrad which utilises a moving average of squared gradients
- [Kingma, Ba (2014) Adam: A Method for Sochastic Optimization](https://arxiv.org/abs/1412.6980)
  - Variant of RMSprop, with addition of momentum-like updates
- [Overview of various optimization methods by Sebastian Ruder](https://ruder.io/optimizing-gradient-descent/index.html#fn14)
  - [arXiv link](https://arxiv.org/abs/1609.04747)