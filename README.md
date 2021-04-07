# Deep Learning Experience

1. **Baseline**. Simple forecasting methods such as persistence and averages.
2. **Autoregression**. The Box-Jenkins process and methods such as SARIMA.
3. **Exponential Smoothing**. Single, double and triple exponential smoothing methods.
4. **Linear Machine Learning**. Linear regression methods and variants such as regularization.
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
* Try topology patterns (fan out then in) and rules of thumb from books and papers.
## Batches and Epochs
The batch size defines the gradient and how often to update weights. An epoch is the entire training data exposed to the network, batch-by-batch.
Some network architectures are more sensitive than others to batch size. I see Multilayer Perceptrons as often robust to batch size, whereas LSTM and CNNs quite sensitive, but that is just anecdotal.
## Common Regularization Methods
1. **Dropout** in machine learning refers to the process of randomly ignoring certain nodes in a layer during training.
**It is proposed that dropout (with p=0.5) better to use on each of the fully connected (dense) layers before the output; it is better to not use on the convolutional layers. This became the most commonly used configuration.
3. **Early stopping:** stop training automatically when a specific performance measure (eg. Validation loss, accuracy) stops improving. It is a kind of cross-validation strategy where we keep one part of the training set as the validation set. When we see that the performance on the validation set is getting worse, we immediately stop the training on the model. This is known as early stopping
4. **Weight decay:** incentivize the network to use smaller weights by adding a penalty to the loss function (this ensures that the norms of the weights are relatively evenly distributed amongst all the weights in the networks, **which prevents just a few weights from heavily influencing network output).**
5. **Noise:** allow some random fluctuations in the data through augmentation (which makes the network robust to a larger distribution of inputs and hence improves generalization).
6. **Model combination:** average the outputs of separately trained neural networks (requires a lot of computational power, data, and time).

## Reducing the variance
A successful approach to reducing the variance of neural network models is to train multiple models instead of a single model and to combine the predictions from these models. This is called **Ensemble Learning** and not only reduces the variance of predictions but also can result in predictions that are better than any single model.
1. Neural network models are nonlinear and have a high variance, which can be frustrating when preparing a final model for making predictions.
2. Ensemble learning combines the predictions from multiple neural network models to reduce the variance of predictions and reduce generalization error.
3. Techniques for ensemble learning can be grouped by the element that is varied, such as training data, the model, and how predictions are combined.
