# adversarial-training-for-Digit-recognition-system
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def preprocess_data(x_train, y_train, x_test, y_test):
    # Reshape and normalize the input data
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

    # One-hot encode the target labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=5):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_data=(x_test, y_test))

def fgsm_attack(image, epsilon, data_grad):
    # Get the sign of the gradient
    sign_data_grad = tf.sign(data_grad)
    # Create the perturbed image by adjusting each pixel along the sign of the gradient
    perturbed_image = image + epsilon * sign_data_grad
    # Clip the perturbed image to ensure it stays within the valid pixel range [0, 1]
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)
    return perturbed_image

def generate_adversarial_examples(model, images, labels, epsilon=0.01):
    # Convert NumPy array to TensorFlow tensor
    images = tf.convert_to_tensor(images)

    # Record gradients of the loss with respect to the input images
    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
    # Get the gradients of the loss with respect to the input images
    gradients = tape.gradient(loss, images)
    # Call the FGSM attack to generate adversarial examples
    adversarial_images = fgsm_attack(images, epsilon, gradients)
    return adversarial_images


def main():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess the data
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)

    # Create and train the model
    model = create_model()
    train_model(model, x_train, y_train, x_test, y_test)

    # Generate adversarial examples for training
    adversarial_x_train = generate_adversarial_examples(model, x_train, y_train)

    # Concatenate the original training data with adversarial examples
    x_train_combined = np.concatenate((x_train, adversarial_x_train))
    y_train_combined = np.concatenate((y_train, y_train))

    # Train the model with adversarial examples
    train_model(model, x_train_combined, y_train_combined, x_test, y_test)

    # Use webcam for digit recognition
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam")
        return

    # Create a window with cv2.WINDOW_GUI_NORMAL flag
    cv2.namedWindow('Webcam', cv2.WINDOW_GUI_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for digit recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (28, 28))
        gray = gray.reshape((1, 28, 28, 1)).astype('float32') / 255.0

        # Make prediction
        prediction = np.argmax(model.predict(gray), axis=-1)
        cv2.putText(frame, f"Prediction: {prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
