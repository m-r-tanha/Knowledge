# Deep Learning Experience

1. **Baseline**. Simple forecasting methods such as persistence and averages.
2. **Autoregression**. The Box-Jenkins process and methods such as SARIMA.
3. **Exponential Smoothing**. Single, double and triple exponential smoothing methods.
4. **Linear Machine Learning**. Linear regression methods and variants such as regularization.
5. **Nonlinear Machine Learning**. kNN, decision trees, support vector regression and
more.
6. **Ensemble Machine Learning**. Random forest, gradient boosting, stacking and more.
7. **Deep Learning**. MLPs, CNNs, LSTMs, and Hybrid models.

**We should compare the performance of various algorithms in terms of their:**
1. Convergence (how fast they reach the answer)
2. Precision (how close do they approximate the exact answer)
3. Robustness (do they perform well for all functions or just a small subset)
4. General performance (e.g. computational complexity)

**During learning is a neural network. It uses randomness in two ways:**

* Random initial weights (model coefficients).
* Random shuffle of samples each epoch.
* Try taking an existing model and retraining a new input and output layer for your problem **(transfer learning)**

## Good configuration for your problem.

* Try one hidden layer with a lot of neurons (wide).
* Try a deep network with few neurons per layer (deep).
* Try combinations of the above.
* Try architectures from recent papers on problems similar to yours.
* Try topology patterns (fan out then in) and rules of thumb from books and papers.

## Callback Function
1. **keras.callbacks.History()** This is automatically included in .fit().
2. **keras.callbacks.ModelCheckpoint** which saves the model with its weights at a certain point in the training. This can prove useful if your model is running for a long time and a system failure happens. Not all is lost then. It's a good practice to save the model weights only when an improvement is observed as measured by the acc.
The EarlyStopping callback will stop training once triggered, but the model at the end of training may not be the model with **best performance on the validation dataset**.
An additional callback is required that will **save the best model observed during training for later use. This is the ModelCheckpoint callback.**
- mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
- es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
- fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0, callbacks=[es, mc])
4. **keras.callbacks.EarlyStopping** stops the training when a monitored quantity has stopped improving.
5. **keras.callbacks.LearningRateScheduler** will change the learning rate during training.


## Batches and Epochs
The batch size defines the gradient and how often to update weights. An epoch is the entire training data exposed to the network, batch-by-batch.
Some network architectures are more sensitive than others to batch size. I see Multilayer Perceptrons as often robust to batch size, whereas LSTM and CNNs quite sensitive, but that is just anecdotal.
## Common Regularization Methods
1. **Dropout** in machine learning refers to the process of randomly ignoring certain nodes in a layer during training.
 **It is proposed that dropout (with p=0.5) better to use on each of the fully connected (dense) layers before the output; it is better to not use on the convolutional layers. This became the most commonly used configuration.**
3. **Early stopping:** stop training automatically when a specific performance measure (eg. Validation loss, accuracy) stops improving. It is a kind of cross-validation strategy where we keep one part of the training set as the validation set. When we see that the performance on the validation set is getting worse, we immediately stop the training on the model. This is known as early stopping
4. **Weight decay:** incentivize the network to use smaller weights by adding a penalty to the loss function (this ensures that the norms of the weights are relatively evenly distributed amongst all the weights in the networks, **which prevents just a few weights from heavily influencing network output).**
5. **Noise:** allow some random fluctuations in the data through augmentation (which makes the network robust to a larger distribution of inputs and hence improves generalization).
6. **Model combination:** average the outputs of separately trained neural networks (requires a lot of computational power, data, and time).

## Reducing the variance
A successful approach to reducing the variance of neural network models is to train multiple models instead of a single model and to combine the predictions from these models. This is called **Ensemble Learning** and not only reduces the variance of predictions but also can result in predictions that are better than any single model.
1. Neural network models are nonlinear and have a high variance, which can be frustrating when preparing a final model for making predictions.
2. Ensemble learning combines the predictions from multiple neural network models to reduce the variance of predictions and reduce generalization error.
3. Techniques for ensemble learning can be grouped by the element that is varied, such as training data, the model, and how predictions are combined.

## In CNN and LSTM # of filter and hidden layer?
1. **CNN:**For this you need to understand what filters does actually.
In every layer filters are there to capture patterns. For example in the **first layer** filters capture patterns like **edges, corners, dots** etc. In the **subsequent layers** we **combine those patterns to make bigger patterns**. Like combine edges to make squares, circle etc.
Now as we move forward in the layers the patterns gets more complex, hence larger combinations of patterns to capture. That's why we increase filter size in the subsequent layers to capture as many combinations as possible.
2. **LSTM:** Ultimately, hidden layers can help improve the accuracy of a model but only up to a certain point. Determining how many hidden layers a model should have is as much an art as a science, and is highly dependent on the type of data you are analyzing.

## ConvLSTM 
It is a type of recurrent neural network for spatio-temporal prediction that has convolutional structures in both the input-to-state and state-to-state transitions. 
  ##### ConvLSTM layer input
The **LSTM** cell input is a set of data over time, that is, a 3D tensor with shape **(samples, time_steps, features)**. The **Convolution** layer input is a set of images as a 4D tensor with shape **(samples, channels, rows, cols)**. The input of a **ConvLSTM** is a set of images over time as a **5D** tensor with shape **(samples, time_steps, channels, rows, cols).**

### Overfitting and Underfitting
![Plot Behav](https://github.com/m-r-tanha/Knowledge/blob/main/VsrRD.png)
![Early Stop](https://github.com/m-r-tanha/Knowledge/blob/main/early%20stopping.png)
- If validation loss >> training loss you can call it overfitting.
- If validation loss  > training loss you can call it some overfitting.
- If validation loss  < training loss you can call it some underfitting.
- If validation loss << training loss you can call it underfitting.

### Evaluation results
- **Mean Absolute Error (MAE):** The average difference between predicted values and true values. This value is based on the same units as the label, in this case dollars. The lower this value is, the better the model is predicting.
**Root Mean Squared Error (RMSE):** The square root of the mean squared difference between predicted and true values. The result is a metric based on the same unit as the label (dollars). When compared to the MAE (above), a larger difference indicates greater variance in the individual errors (for example, with some errors being very small, while others are large).
**Relative Squared Error (RSE):** A relative metric between 0 and 1 based on the square of the differences between predicted and true values. The closer to 0 this metric is, the better the model is performing. Because this metric is relative, it can be used to compare models where the labels are in different units.
**Relative Absolute Error (RAE):** A relative metric between 0 and 1 based on the absolute differences between predicted and true values. The closer to 0 this metric is, the better the model is performing. Like RSE, this metric can be used to compare models where the labels are in different units.
**Coefficient of Determination (R2):** This metric is more commonly referred to as R-Squared, and summarizes how much of the variance between predicted and true values is explained by the model. The closer to 1 this value is, the better the model is performing.
