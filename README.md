# Lymphoma-Subtype-Classification-DenseNet-CNN

This code defines and evaluates two convolutional neural network (CNN) architectures — a custom implementation of **DensNet-121** and a custom **CNN** model — to classify images from a lymphoma dataset. Here’s a breakdown of the main components and what the code does:

### Overview
- **Purpose**: The code trains two deep learning models for lymphoma cancer classification using histopathological images from a dataset containing three classes: Chronic Lymphocytic Leukemia (CLL), Follicular Lymphoma (FL), and Mantle Cell Lymphoma (MCL).

The code can be broken down into the following sections:

### 1. Data Loading and Preprocessing
**Key Functions and Steps**:
- **`load_data` Function**: Loads the images from the specified directory and labels them based on the class they belong to. Labels are converted to one-hot encoded vectors using `to_categorical`.
  - **Classes**: The dataset is divided into three classes: `CLL`, `FL`, and `MCL`.
  - **Data Count**: Prints the count of images for each class.

- **Splitting the Dataset**: The images and labels are split into training, validation, and test datasets using `split_data` with a ratio of 5% for validation and testing.

- **`preprocess_image` Function**: Converts TIFF images to RGB, resizes them to `640x640`, and normalizes pixel values between `0` and `1`.

- **Dataset Creation**:
  - **`create_dataset` Function**: This function creates a TensorFlow dataset pipeline.
  - **Caching and Prefetching**: Utilizes caching and prefetching to improve training performance.
  - **Batch Size**: Defined as 8.

### 2. Utility Functions
- **`plot_image`**: Plots a sample image.
- **`label_decoder`**: Converts one-hot encoded labels back to their corresponding class names.
- **Visualization**: The `plot_loss_curve` and `plot_accuracy_curve` functions help visualize the loss and accuracy during training.

### 3. Early Stopping and Callbacks
- **Custom Early Stopping Class**: Monitors a specified metric and stops training based on either reaching a max epoch limit or when the monitored metric remains above a threshold for a specified patience.
- **`TotalTimeMemoryCallback`**: Custom callback to track memory usage and training time.
- **ModelCheckpoint**: Saves the best-performing model based on validation accuracy.

### 4. First Model - DenseNet-121
**DenseNet-121 Architecture**:
- **DenseNet-121** is a pre-built model used in medical image classification. This custom implementation uses repeated **dense blocks**, each of which contains **convolutional** and **batch normalization** layers.
- **`__conv_block`, `__dense_block`, `__transition_block` Functions**:
  - **Convolutional Block**: Applies two convolution operations to expand and then reduce the feature maps.
  - **Dense Block**: Repeatedly applies the convolution block to create dense connections.
  - **Transition Block**: Reduces feature maps and downscales the image.

The **DenseNet-121** model is then trained for a maximum of 60 epochs using the Adam optimizer. Metrics used are `accuracy`, `precision`, and `recall`. The training process is tracked using early stopping, model checkpoints, and the `TotalTimeMemoryCallback`.

**Training DenseNet-121**:
- The history of training is visualized using `plot_loss_curve` and `plot_accuracy_curve`.
- The model is evaluated using a **confusion matrix** and the performance metrics (accuracy, precision, and recall) are printed.

### 5. Second Model - Custom CNN
**Custom CNN Architecture**:
- This model is a simpler, custom-designed CNN.
- **Architecture Details**:
  - **Multiple Convolution Layers**: Six convolution layers with increasing filter sizes (32 to 1024), each followed by **batch normalization**, **activation**, and **max pooling** layers.
  - **Dense Layers**: The model uses fully connected (Dense) layers after global average pooling to produce the final output. Dropout is applied to prevent overfitting.

**Training Custom CNN**:
- Similar to DenseNet-121, the custom CNN is trained for up to 50 epochs.
- Training, validation, and evaluation are done in the same manner as DenseNet-121, with callbacks (`ModelCheckpoint`, `EarlyStopping`, `TotalTimeMemoryCallback`) ensuring efficient training.
- Metrics (accuracy, precision, recall) are plotted, and a confusion matrix is generated for evaluation.

### 6. Explanation of Training Process
**Callbacks Used**:
- **ModelCheckpoint**: Saves the best model based on validation accuracy.
- **EarlyStopping**: Stops training if validation accuracy does not improve for a specified number of epochs or if a certain threshold is met.
- **TotalTimeMemoryCallback**: Tracks and reports the memory usage and time taken during each epoch.

**Metrics and Evaluation**:
- **Plotting Training Curves**: Functions such as `plot_loss_curve` and `plot_accuracy_curve` help track how the model's loss and accuracy evolve during training.
- **Confusion Matrix**: Used to visualize the performance of the model on test data, showing how well the model predicts each class.
  
### 7. Performance Evaluation
Both models are evaluated using:
- **Test Loss, Accuracy, Precision, and Recall**: These metrics help determine how well the models perform on unseen data.
- **Confusion Matrix**: This is used to see where the models are making correct and incorrect predictions.
