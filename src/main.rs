use std::f64;

// Simple neural network layer
#[derive(Debug, Clone)]
pub struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Initialize weights with small random values
        let mut weights = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            let mut row = Vec::with_capacity(input_size);
            for _ in 0..input_size {
                row.push((rand::random::<f64>() - 0.5) * 0.1);
            }
            weights.push(row);
        }
        
        let biases = vec![0.0; output_size];
        
        Layer { weights, biases }
    }
    
    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        let mut outputs = Vec::with_capacity(self.weights.len());
        
        for (i, weight_row) in self.weights.iter().enumerate() {
            let mut sum = self.biases[i];
            for (j, &input) in inputs.iter().enumerate() {
                sum += weight_row[j] * input;
            }
            outputs.push(relu(sum));
        }
        
        outputs
    }
}

// Simple neural network
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        
        NeuralNetwork { layers }
    }
    
    pub fn predict(&self, mut inputs: Vec<f64>) -> Vec<f64> {
        for layer in &self.layers {
            inputs = layer.forward(&inputs);
        }
        inputs
    }
    
    pub fn train_step(&mut self, inputs: &[f64], targets: &[f64], learning_rate: f64) {
        // Forward pass
        let mut layer_outputs = vec![inputs.to_vec()];
        
        for layer in &self.layers {
            let output = layer.forward(layer_outputs.last().unwrap());
            layer_outputs.push(output);
        }
        
        // Backward pass (simplified gradient descent)
        let final_output = layer_outputs.last().unwrap();
        let mut errors: Vec<f64> = final_output.iter()
            .zip(targets.iter())
            .map(|(pred, target)| target - pred)
            .collect();
        
        // Update weights (simplified backpropagation)
        for (layer_idx, layer) in self.layers.iter_mut().enumerate().rev() {
            let layer_input = &layer_outputs[layer_idx];
            
            for (i, error) in errors.iter().enumerate() {
                for (j, &input) in layer_input.iter().enumerate() {
                    layer.weights[i][j] += learning_rate * error * input;
                }
                layer.biases[i] += learning_rate * error;
            }
            
            // Calculate errors for previous layer (simplified)
            if layer_idx > 0 {
                let mut prev_errors = vec![0.0; layer_input.len()];
                for (i, error) in errors.iter().enumerate() {
                    for (j, prev_error) in prev_errors.iter_mut().enumerate() {
                        *prev_error += error * layer.weights[i][j];
                    }
                }
                errors = prev_errors;
            }
        }
    }
}

// Activation functions
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Matrix operations for linear algebra
pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![vec![0.0; cols]; rows],
            rows,
            cols,
        }
    }
    
    pub fn from_vec(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        Matrix { data, rows, cols }
    }
    
    pub fn multiply(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        if self.cols != other.rows {
            return Err("Matrix dimensions don't match for multiplication");
        }
        
        let mut result = Matrix::new(self.rows, other.cols);
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        
        Ok(result)
    }
    
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        
        result
    }
}

// Example usage and training
fn main() {
    // Create a simple 2-3-1 network (2 inputs, 3 hidden, 1 output)
    let mut network = NeuralNetwork::new(&[2, 3, 1]);
    
    // Training data for XOR problem
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    
    println!("Training neural network on XOR problem...");
    
    // Training loop
    for epoch in 0..1000 {
        let mut total_error = 0.0;
        
        for (inputs, targets) in &training_data {
            let prediction = network.predict(inputs.clone());
            let error = (targets[0] - prediction[0]).powi(2);
            total_error += error;
            
            network.train_step(inputs, targets, 0.1);
        }
        
        if epoch % 200 == 0 {
            println!("Epoch {}: Average error = {:.6}", epoch, total_error / training_data.len() as f64);
        }
    }
    
    // Test the trained network
    println!("\nTesting trained network:");
    for (inputs, expected) in &training_data {
        let prediction = network.predict(inputs.clone());
        println!("Input: {:?} -> Prediction: {:.3}, Expected: {:.1}", 
                inputs, prediction[0], expected[0]);
    }
    
    // Matrix operations example
    println!("\nMatrix operations example:");
    let matrix_a = Matrix::from_vec(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ]);
    
    let matrix_b = Matrix::from_vec(vec![
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ]);
    
    match matrix_a.multiply(&matrix_b) {
        Ok(result) => {
            println!("Matrix multiplication result:");
            for row in &result.data {
                println!("{:?}", row);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}
