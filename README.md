# Deep Learning Experience

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
