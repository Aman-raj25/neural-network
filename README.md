# Neural Network with Sigmoid Activation

## Repository: neural-network-sigmoid

### Description
This repository contains an implementation of a simple neural network from scratch using NumPy. The neural network is designed to solve the XOR problem using a single hidden layer with sigmoid activation.

### File Structure
- **main.py**: Contains the implementation of the neural network, including forward propagation, backward propagation, training, and visualization.

### Features
- Implements a neural network with:
  - A customizable number of hidden neurons
  - Sigmoid activation function
  - Gradient descent optimization
  - Adaptive learning rate with decay
- Trains on the XOR dataset
- Visualizes the decision boundary

### Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Aman-raj25/neural-network-sigmoid.git
   cd neural-network-sigmoid
   ```
2. Install dependencies (if required):
   ```bash
   pip install numpy matplotlib
   ```
3. Run the neural network:
   ```bash
   python main.py
   ```

### Expected Output
- The neural network should train for 5000 epochs, printing the loss at every 100 epochs.
- A final accuracy score will be displayed.
- A decision boundary plot will be shown, visualizing the learned function.

### Customization
- Modify the `hidden_size` parameter in `main.py` to change the number of neurons in the hidden layer.
- Adjust the `learning_rate` and `decay_rate` for different training behaviors.
- Increase `epochs` for longer training sessions.

### Author
- **Aman Raj**

