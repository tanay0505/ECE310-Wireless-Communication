After 100 epoch, accuracy = 49.7%

Data Preparation:

The code traverses through a directory structure to find .mat files, extracting data labeled as velocity_spectrum_ro from them.
It pads or truncates the data to ensure a fixed number of time steps (fixed_time_steps) and checks for empty files or invalid data.
The code determines the number of classes based on the labels found in the file names, and then loads the data and labels, ensuring they are consistent.
The data is normalized, and the labels are one-hot encoded.
Model Architecture:

A Sequential model is created with:
A Flatten layer to transform the data into a 1D array.
Two Dense layers with 128 and 64 units respectively, both using ReLU activation.
Two Dropout layers for regularization, which help prevent overfitting.
An output layer with a softmax activation to predict the class probabilities.
The model is compiled using the Adam optimizer, categorical cross-entropy loss, and accuracy as a metric.
Training Configuration:

The data is split into training and testing sets.
A custom data generator is used to feed batches of data into the model during training, which can be useful for handling larger datasets.
Early stopping is implemented to stop training when the validation loss does not improve for three consecutive epochs, helping to prevent overfitting.
Training the Model:

The model is trained using the generator function, with training and validation accuracy/loss being tracked.
Training history is plotted to show the model's performance over epochs.
Evaluation:

The model's accuracy is calculated on the test set.
A confusion matrix is generated to visualize the true vs. predicted labels, providing insights into the model's performance.
Plotting Accuracy and Loss:

Training and validation accuracy, as well as loss, are plotted to assess how well the model performs over time.
Confusion Matrix Plotting:

The confusion matrix helps to evaluate the model’s classification accuracy by showing the frequency of correct and incorrect predictions.
