# 41.72 for 10epoch

import os
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Dropout, TimeDistributed, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Constants
DATA_DIRECTORY = r"C:/Users/shrut/OneDrive/Desktop/Engineering/Mechanical Engineering/Sem 5/Wireless Communications/BVP"
SAMPLE_SIZE = 20
FIXED_TIME_STEPS = 50
NUM_CLASSES = 11
BATCH_SIZE = 100
EPOCHS = 10  # Limited to 10 epochs

# Padding or truncation function
def pad_or_truncate(data, target_length):
    if data.shape[2] > target_length:
        return data[:, :, :target_length]
    elif data.shape[2] < target_length:
        pad_width = target_length - data.shape[2]
        return np.pad(data, ((0, 0), (0, 0), (0, pad_width)), 'constant')
    return data

# Load data function
def load_data(data_directory):
    all_data, labels = [], []
    for root, dirs, files in os.walk(data_directory):
        for dir_name in dirs:
            if "6-link" in dir_name:
                user_folder_path = os.path.join(root, dir_name)
                for user_folder in os.listdir(user_folder_path):
                    user_path = os.path.join(user_folder_path, user_folder)
                    if os.path.isdir(user_path) and user_folder.startswith("user"):
                        for file in os.listdir(user_path):
                            if file.endswith('.mat'):
                                file_path = os.path.join(user_path, file)
                                if os.path.getsize(file_path) == 0:
                                    continue
                                try:
                                    mat_data = scipy.io.loadmat(file_path)
                                    if 'velocity_spectrum_ro' in mat_data:
                                        bvp_data = mat_data['velocity_spectrum_ro']
                                        if bvp_data.shape[0:2] == (SAMPLE_SIZE, SAMPLE_SIZE):
                                            bvp_data = pad_or_truncate(bvp_data, FIXED_TIME_STEPS)
                                            label = int(file.split('-')[1])
                                            if label < NUM_CLASSES:
                                                all_data.append(bvp_data)
                                                labels.append(label)
                                except Exception as e:
                                    print(f"Error loading file {file_path}: {e}")

    return np.array(all_data), np.array(labels)

# Load BVP data
X, y = load_data(DATA_DIRECTORY)
X = X.astype('float32') / 2.0  # Normalize
y = to_categorical(y, NUM_CLASSES)  # One-hot encode

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Simplified CNN + LSTM
model = Sequential()

# CNN layers for spatial feature extraction
model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'), input_shape=(FIXED_TIME_STEPS, SAMPLE_SIZE, SAMPLE_SIZE, 1)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))

# LSTM layers for temporal feature extraction
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))

# Dense output layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

# Train model for 10 epochs
history = model.fit(
    X_train.reshape(-1, FIXED_TIME_STEPS, SAMPLE_SIZE, SAMPLE_SIZE, 1),
    y_train,
    epochs=EPOCHS,  # Only 10 epochs
    batch_size=BATCH_SIZE,
    validation_data=(X_test.reshape(-1, FIXED_TIME_STEPS, SAMPLE_SIZE, SAMPLE_SIZE, 1), y_test),
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test.reshape(-1, FIXED_TIME_STEPS, SAMPLE_SIZE, SAMPLE_SIZE, 1), y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Confusion matrix
y_pred = model.predict(X_test.reshape(-1, FIXED_TIME_STEPS, SAMPLE_SIZE, SAMPLE_SIZE, 1))
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.3f})')
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.show()
