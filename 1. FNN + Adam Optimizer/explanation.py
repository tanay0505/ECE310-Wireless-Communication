51.34%

The code is using a basic feedforward neural network (FNN) or fully connected neural network (FCNN) for classification, implemented with Keras (part of TensorFlow).

The model uses the Adam optimizer and categorical crossentropy loss, which is standard for multi-class classification tasks.
                                                                                                                                                
Data Preparation:
The script loads Biological Vital Parameter (BVP) data from .mat files located in a specified directory.
It pads or truncates each data sample to ensure a fixed number of time steps (fixed_time_steps).
The labels are extracted from filenames and one-hot encoded for multi-class classification.
Data Splitting:
The data is normalized and split into training and testing sets using an 80-20 split ratio.

                                                           
Model Architecture:
A basic feedforward neural network is used:
Flatten layer to convert 3D data into a 1D format.
Two Dense layers with 128 and 64 neurons, using ReLU activation.
Dropout layers (with 0.5 dropout rate) for regularization.
An Output layer with num_classes neurons and softmax activation for classification.

                                                               
Training Process:
Early stopping is used to monitor validation loss and prevent overfitting.
A custom data generator is implemented for memory-efficient training in batches.
The model is compiled with the Adam optimizer and categorical crossentropy loss.
Evaluation:

The model's performance is evaluated on the test set, and the final accuracy is displayed.
