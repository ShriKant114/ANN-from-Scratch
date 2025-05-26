#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// Activation Functions
double relu(double x) {
    return x > 0 ? x : 0;
}
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Normalization helpers
double normalize_input(double x) {
    return x / 1500.0;  // max input size
}
double normalize_output(double y) {
    return y / 100.0;   // max price
}
double denormalize_output(double y) {
    return y * 100.0;
}

// Small random weight
double rand_weight() {
    return ((double) rand() / RAND_MAX) * 0.2 - 0.1;  // -0.1 to 0.1
}

class SimpleANN {
public:
    double w1[4]; // input → hidden
    double b1[4];
    double w2[4]; // hidden → output
    double b2;
    double lr; // learning rate

    SimpleANN(double learning_rate = 0.01) {
        srand(time(0));
        for (int i = 0; i < 4; i++) {
            w1[i] = rand_weight();
            b1[i] = rand_weight();
            w2[i] = rand_weight();
        }
        b2 = rand_weight();
        lr = learning_rate;
    }

    // Forward pass
    double forward(double x, double h[4]) {
        for (int i = 0; i < 4; i++) {
            h[i] = relu(w1[i] * x + b1[i]);
        }
        double out = b2;
        for (int i = 0; i < 4; i++) {
            out += w2[i] * h[i];  // linear output
        }
        return out;
    }

    // Train on all data for N epochs
    void train(vector<double>& X, vector<double>& Y, int epochs) {
        for (int e = 0; e < epochs; e++) {
            double total_loss = 0;
            for (size_t i = 0; i < X.size(); i++) {
                double h[4];
                double x = X[i];
                double y = Y[i];

                // Forward
                double out = forward(x, h);

                // Loss = (y - out)^2
                double loss = (y - out) * (y - out);
                total_loss += loss;

                // Backpropagation
                double dL_dout = -2 * (y - out);

                for (int j = 0; j < 4; j++) {
                    // Output weights update
                    double d_out_d_w2 = h[j];
                    w2[j] -= lr * dL_dout * d_out_d_w2;

                    // Hidden weights update
                    double d_h_j = relu_derivative(w1[j] * x + b1[j]);
                    w1[j] -= lr * dL_dout * w2[j] * d_h_j * x;
                    b1[j] -= lr * dL_dout * w2[j] * d_h_j;
                }

                b2 -= lr * dL_dout;
            }

            if (e % 1000 == 0)
                cout << "Epoch " << e << " Loss: " << total_loss / X.size() << endl;
        }
    }

    double predict(double x) {
        double h[4];
        return forward(x, h);
    }
};

int main() {
    // Raw training data
    vector<double> sizes = {500, 750, 1000, 1200, 1500};
    vector<double> prices = {35, 35, 50, 60, 75};

    // Normalize data
    for (int i = 0; i < sizes.size(); i++) {
        sizes[i] = normalize_input(sizes[i]);
        prices[i] = normalize_output(prices[i]);
    }

    // Initialize model
    SimpleANN model(0.01); // Learning rate

    // Train the model
    model.train(sizes, prices, 20000);  // 20K epochs

    // Predict
    double test_input;
    cout << "\nEnter home size (sqft): ";
    cin >> test_input;

    double norm_input = normalize_input(test_input);
    double prediction = model.predict(norm_input);
    double final_price = denormalize_output(prediction);

    cout << "Predicted price (lakhs): ₹" << final_price << endl;

    return 0;
}
