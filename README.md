# Deep Learning Experience

1. **Baseline**. Simple forecasting methods such as persistence and averages.
2. **Autoregression**. The Box-Jenkins process and methods such as SARIMA.
3. **Exponential Smoothing**. Single, double and triple exponential smoothing methods.
4. **Linear Machine Learning**. Linear regression methods and variants such as regularization.
5. **Nonlinear Machine Learning**. kNN, decision trees, support vector regression and
more.
6. **Ensemble Machine Learning**. Random forest, gradient boosting, stacking and more.
7. **Deep Learning**. MLPs, CNNs, LSTMs, and Hybrid models.


During learning is a neural network. It uses randomness in two ways:

* Random initial weights (model coefficients).
* Random shuffle of samples each epoch.
* Try taking an existing model and retraining a new input and output layer for your problem **(transfer learning)**

## Good configuration for your problem.

* Try one hidden layer with a lot of neurons (wide).
* Try a deep network with few neurons per layer (deep).
* Try combinations of the above.
* Try architectures from recent papers on problems similar to yours.
* Try topology patterns (fan out then in) and rules of thumb from books and papers (see links below).
## Batches and Epochs
The batch size defines the gradient and how often to update weights. An epoch is the entire training data exposed to the network, batch-by-batch.
Some network architectures are more sensitive than others to batch size. I see Multilayer Perceptrons as often robust to batch size, whereas LSTM and CNNs quite sensitive, but that is just anecdotal.
## Common Regularization Methods
1. **Early stopping:** stop training automatically when a specific performance measure (eg. Validation loss, accuracy) stops improving
2. **Weight decay:** incentivize the network to use smaller weights by adding a penalty to the loss function (this ensures that the norms of the weights are relatively evenly distributed amongst all the weights in the networks, **which prevents just a few weights from heavily influencing network output).**
3. **Noise:** allow some random fluctuations in the data through augmentation (which makes the network robust to a larger distribution of inputs and hence improves generalization).
4. **Model combination:** average the outputs of separately trained neural networks (requires a lot of computational power, data, and time).
