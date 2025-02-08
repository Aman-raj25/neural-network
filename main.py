import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5, decay_rate=0.001):
        self.learning_rate = learning_rate  # Initial learning rate
        self.initial_lr = learning_rate  # Store initial learning rate for decay
        self.decay_rate = decay_rate  # Decay rate for adaptive learning
        
        # Initialize weights and biases with small random values
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))  # Weights for input to hidden layer
        self.bias_hidden = np.zeros((1, hidden_size))  # Bias for hidden layer
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))  # Weights for hidden to output layer
        self.bias_output = np.zeros((1, output_size))  # Bias for output layer
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  # Sigmoid activation function
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)  # Derivative of sigmoid function
    
    def forward_propagation(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden  # Compute hidden layer input
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)  # Apply activation function
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output  # Compute output layer input
        self.output_layer_output = self.sigmoid(self.output_layer_input)  # Apply activation function
        return self.output_layer_output  # Return final output
    
    def compute_loss(self, Y, predictions):
        m = Y.shape[0]  # Number of training examples
        return -np.sum(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions)) / m  # Binary cross-entropy loss
    
    def backward_propagation(self, X, Y):
        m = X.shape[0]  # Number of training examples
        
        # Compute gradients for output layer
        output_error = self.output_layer_output - Y  # Derivative of loss w.r.t output activation
        output_gradient_weights = np.dot(self.hidden_layer_output.T, output_error) / m  # Gradient of hidden-output weights
        output_gradient_bias = np.sum(output_error, axis=0, keepdims=True) / m  # Gradient of output bias
        
        # Compute gradients for hidden layer
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_layer_output)  # Backpropagation to hidden layer
        hidden_gradient_weights = np.dot(X.T, hidden_error) / m  # Gradient of input-hidden weights
        hidden_gradient_bias = np.sum(hidden_error, axis=0, keepdims=True) / m  # Gradient of hidden bias
        
        # Update weights and biases using gradient descent
        self.weights_input_hidden -= self.learning_rate * hidden_gradient_weights
        self.bias_hidden -= self.learning_rate * hidden_gradient_bias
        self.weights_hidden_output -= self.learning_rate * output_gradient_weights
        self.bias_output -= self.learning_rate * output_gradient_bias
    
    def train(self, X, Y, epochs=5000):
        losses = []  # Store loss values for visualization
        for epoch in range(epochs):
            self.learning_rate = self.initial_lr / (1 + self.decay_rate * epoch)  # Adaptive learning rate
            predictions = self.forward_propagation(X)  # Forward pass
            loss = self.compute_loss(Y, predictions)  # Compute loss
            self.backward_propagation(X, Y)  # Backward pass (gradient descent)
            losses.append(loss)  # Store loss for visualization
            
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")
        
        # Plot loss curve to visualize training progress
        plt.plot(range(epochs), losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve for Training")
        plt.show()
    
    def predict(self, X):
        predictions = self.forward_propagation(X)  # Perform forward propagation
        return (predictions > 0.5).astype(int)  # Convert probabilities to binary output
    
    def accuracy(self, X, Y):
        predictions = self.predict(X)  # Get predictions
        return np.mean(predictions == Y) * 100  # Compute accuracy
    
    def visualize_decision_boundary(self, X, Y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = self.predict(grid).reshape(xx.shape)
        
        plt.contourf(xx, yy, preds, alpha=0.3, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap=plt.cm.Spectral, edgecolors='k')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Decision Boundary of Neural Network")
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)  # Set seed for reproducibility
    
    # XOR gate dataset (non-linearly separable problem)
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data
    Y_train = np.array([[0], [1], [1], [0]])  # Expected output
    
    # Initialize neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5, decay_rate=0.001)
    
    # Train the neural network
    nn.train(X_train, Y_train, epochs=5000)
    
    # Make predictions on the training data
    predictions = nn.predict(X_train)
    print("Predictions:", predictions)  # Display predicted outputs
    
    # Calculate accuracy
    accuracy = nn.accuracy(X_train, Y_train)
    print(f"Training Accuracy: {accuracy:.2f}%")
    
    # Visualize decision boundary
    nn.visualize_decision_boundary(X_train, Y_train)
