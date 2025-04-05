import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Create an equivalent train_test_split
def train_test_split(X, y, train_size, random_state, shuffle=True):

    if not (0 < train_size <= 1):
        raise ValueError("train_size must be between 0 and 1.")
    
    # Set training size
    train_size = int(X.shape[0] * train_size) 

    # Shuffle the data or not
    if shuffle:
        # Set the seed for the randomizer
        np.random.seed(int(random_state))
        # Randomize the indices for the dataset
        indices = np.random.permutation(X.shape[0]) # Randomize rows
    
    else:
        indices = np.arange(X.shape[0])

    # Access rows. Note these are NumPy arrays!
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # If input X, y is a pandas df
    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[train_indices, :], X.iloc[test_indices, :]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    # If input X, y is assumed as a numpy array
    else:
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]

        # Ensure y_train and y_test retain 1D shape if y is originally 1D
        if y.ndim == 1:
            y_train, y_test = y_train.flatten(), y_test.flatten()

    return X_train, X_test, y_train, y_test


def k_fold(X, n_splits, shuffle, random_state):
    
    # Store dictionary of test and train indices per fold
    folds = []
    
    # Shuffle the data
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(X.shape[0])
    
    else:
        indices = np.arange(X.shape[0])

    # Compute the base validation size and distribute extra samples
    base_validation_size = X.shape[0] // n_splits
    extra_samples = X.shape[0] % n_splits
    
    # Set starting point for the indices and increment it per fold
    starting_index = 0

    for fold in range(n_splits):
        
        # Compute actual validation size for this fold
        validation_size = base_validation_size + (1 if fold < extra_samples else 0)

        # Specify which indices to use for training and validation
        validation_indices = indices[starting_index:starting_index + validation_size]
        training_indices = np.concatenate((indices[:starting_index], indices[starting_index + validation_size:]))

        # Create a dictionary of indices for train and validation
        train_valid_dict = {
            "Training indices" : training_indices,
            "Validation indices" : validation_indices
        }
        
        # Append the created dictionary
        folds.append(train_valid_dict)
        
        # Increment starting index to a new starting index
        starting_index += validation_size

    return folds

# Define sigmoid function
def sigmoid_func(z):
    return 1 / (1 + np.exp(-z)) # Shape: (m, 1)

# Assume of the form y = w1X1 + w2X2 +...+ WnXn for now
def decision_boundary_func(w, b, X):
    return X @ w + b # Shape: (m, n)

def fit(X, y, no_iterations, _lambda, learning_rate=0.001, random_state=1, verbose=False, show_loss_plot=False, return_final_cost=False, epsilon=1e-6):

    """
    Params:
    X (pandas dataframe) shape: (m, n) - features
    y (pandas dataframe) shape: (m, 1) - target
    """

    # Define logistic cost function
    def logistic_cost(y_pred, y, _lambda, w, m):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -(1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) 
        reg_term = (_lambda / (2 * m)) * np.sum(w ** 2) # Shape: (m, 1)
        return loss + reg_term

    # Set random seed
    np.random.seed(random_state)

    # Set m
    m = X.shape[0] # Number of rows

    # Convert X to numpy
    if isinstance(X, pd.DataFrame):
        X_np = X.to_numpy() # Shape: (m, n)
        y_np = y.to_numpy().reshape(m, 1) # Shape: (m, 1)
    
    # Case if X is already a numpy array
    else:
        X_np = X
        y_np = y.reshape(m, 1)
    
    # Initialize tmp_w and tmp_b values
    tmp_w = np.random.randn(X.shape[1], 1) # Shape: (n, 1)
    tmp_b = 0

    cost_list = []

    for iter in range(no_iterations):

        # Compute predictions
        z = decision_boundary_func(tmp_w, tmp_b, X_np) # Shape (m, 1)
        y_pred = sigmoid_func(z) # Shape: (m, 1)
      
        # Calculate gradients
        # X_np.T shape (n, m)
        grad_w = (1 / m) * (X_np.T @ (y_pred - y_np)) + (_lambda / m) * tmp_w # Shape: (n, 1)
        grad_b = (1 / m) * np.sum(y_pred - y_np) # Scalar

        # Simultaneously update tmp_w and tmp_b based on the gradient descent function
        # and learning rate
        tmp_w -= learning_rate * grad_w # Shape: (n, 1)
        tmp_b -= learning_rate * grad_b # Shape: (n, 1)

        # Update the cost based on the updated vals 
        # of tmp_w and tmp_b
        tmp_cost = logistic_cost(y_pred, y_np, _lambda, tmp_w, m=X_np.shape[0])
        cost_list.append(tmp_cost)

        if verbose:
            # Progress report
            if iter % 10 == 0:
                print(f"w: {tmp_w.T}, b: {tmp_b:.3f}, cost: {tmp_cost:.3f}")
        
        # Terminate the loop if cost_list[iter] - cost_list[iter - 1] < epsilon
        # Begin process at the iteration one for you to have a comparison to the previous
        if iter >= 1 and abs(cost_list[-1] - cost_list[-2]) < epsilon:
            if verbose:
                print(f"Exited loop at {iter} as the change in cost is {abs(cost_list[-1] - cost_list[-2])}")
                break
            else:
                break
            
    if show_loss_plot:
        # Create a data for the cost vs iteration
        cost_df = pd.DataFrame({
        "Iteration": np.arange(len(cost_list)),
        "Cost": cost_list
        })
        sns.lineplot(data=cost_df, x="Iteration", y="Cost")
    
    if return_final_cost:
        return tmp_w, tmp_b, cost_list[-1]
    else:
        return tmp_w, tmp_b


def optimize_lambda(lambda_range, X, y, no_iterations, n_splits, random_state, log_scale= True, shuffle=True, show_loss_plot=False):
    
    folds = k_fold(X, n_splits, shuffle, random_state)

    avg_lam_loss = []

    for lam in lambda_range:

        fold_loss = []

        for split, fold in enumerate(folds):
            train_idx = fold['Training indices']
            val_idx = fold['Validation indices']

            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]

            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            w, b = fit(X_tr, y_tr, no_iterations, _lambda=lam, verbose=False, show_loss_plot = False)
        
            # Predicting probabilities
            z_pred = decision_boundary_func(w, b, X_val)
            g_z_pred = sigmoid_func(z_pred) # Convert to probabilities       
            g_z_pred = np.clip(g_z_pred, 1e-15, 1 - 1e-15) # avoid log 0 
            # Determine average loss per lambda   
            loss = -np.mean(y_val.to_numpy().reshape(-1,1) * np.log(g_z_pred) + (1 - y_val.to_numpy().reshape(-1, 1)) * np.log(1 - g_z_pred))
            fold_loss.append(loss)

        # Append average loss per lambda value
        avg_lam_loss.append(np.mean(fold_loss))
        print(f"Lambda: {lam:.4f} | Avg Log Loss: {np.mean(fold_loss):.4f}")

    best_lambda = lambda_range[np.argmin(avg_lam_loss)]
    if show_loss_plot:
        # plot x vals lambda range
        x = lambda_range.flatten()
        y = np.array(avg_lam_loss).flatten()
        plt.plot(x, y)
        plt.xlabel('lambda (Regularization strength)')
        plt.ylabel('loss')
        plt.grid(True)
        plt.title('lambda vs log loss')
        if log_scale:
            plt.xscale('log')
        plt.show()

    return f"Best lambda: {best_lambda:.4f}, Average validation Error: {np.min(avg_lam_loss)}"


def determine_optimal_iterations(iterations, X, y, _lambda, show_plot=True):
    # Final cost estimates
    end_costs = []
    for iteration in iterations:
        w, b, _cost = fit(X, y, iteration, _lambda, return_final_cost=True)
        end_costs.append(_cost)
    # Determine optimal num_iterations
    optimal_num_iterations = iterations[np.argmin(end_costs)]
    if show_plot:
        data = np.column_stack((iterations, end_costs))
        iteration_end_cost = pd.DataFrame(data, columns=['Iterations', 'End Cost'])
        plt.title('Number of iterations vs end cost')
        sns.lineplot(data=iteration_end_cost, x='Iterations', y='End Cost')

    return f"Optimal number of iterations: {optimal_num_iterations}"    