import os
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


data_directory = "C:/Users/shrut/OneDrive/Desktop/Engineering/Mechanical Engineering/Sem 6/Wireless Communications/BVP"
sample_size = 20 
fixed_time_steps = 50  

def pad_or_truncate(data, target_length):
    if data.shape[2] > target_length:
        # Truncate
        return data[:, :, :target_length]
    elif data.shape[2] < target_length:
        # Pad with zeros
        pad_width = target_length - data.shape[2]
        return np.pad(data, ((0, 0), (0, 0), (0, pad_width)), 'constant')
    else:
        return data

def load_and_stack_bvp_data(data_directory):
    all_data = []
    labels = []
    
    # Iterate through all date folders
    for root, dirs, files in os.walk(data_directory):
        for dir_name in dirs:
            if "6-link" in dir_name:
                user_folder_path = os.path.join(root, dir_name)
                
                # Iterate through useri folders in "6-link"
                for user_folder in os.listdir(user_folder_path):
                    user_path = os.path.join(user_folder_path, user_folder)
                    
                    if os.path.isdir(user_path) and user_folder.startswith("user"):
                        # Load .mat files from user folders
                        for file in os.listdir(user_path):
                            if file.endswith('.mat'):
                                try:
                                    mat_data = scipy.io.loadmat(os.path.join(user_path, file))
                                    
                                    if 'velocity_spectrum_ro' in mat_data:
                                        bvp_data = mat_data['velocity_spectrum_ro']
                                        
                                        if bvp_data.shape[0:2] == (sample_size, sample_size):
                                            bvp_data = pad_or_truncate(bvp_data, fixed_time_steps)
                                            
                                            all_data.append(bvp_data)
                                            
                                            label = int(file.split('-')[1]) 
                                            labels.append(label)
                                        else:
                                            print(f"Skipping file {file} due to shape mismatch: {bvp_data.shape}")
                                    else:
                                        print(f"'velocity_spectrum_ro' not found in file {file}")
                                except Exception as e:
                                    print(f"Error loading file {file}: {e}")
    
    all_data = np.stack(all_data, axis=0)
    labels = np.array(labels)
    
    return all_data, labels

X, y = load_and_stack_bvp_data(data_directory)

print("Unique labels in y before to_categorical:", np.unique(y))

if len(X) == 0 or len(y) == 0:
    raise ValueError("No valid data found. Please check your dataset.")

num_classes = np.max(y) + 1 
print(f"Number of classes: {num_classes}")

X = X.astype('float32') / 2.0  
y = to_categorical(y, num_classes) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([ Flatten(input_shape=(sample_size, sample_size, fixed_time_steps)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy*100:.2f}%')
