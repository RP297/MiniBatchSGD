# MiniBatchSGD
Simple python implementation of a Linear Regression model with MiniBatch SGD.

The script generates a random matrix $A\in \mathbb R^{100 \times 10}$ and a vector $b\in\mathbb R^{100}$ and solves $\min_{x\in\mathbb R^{10}} \||Ax - b\||_2^2$ using Minibatch SGD. Minibatch SGD works by sampling k data points at each iteration, using those to estimate the gradient of $\||Ax-b\||^2_2$, and then takes a gradient step using a stepsize $\eta$.
