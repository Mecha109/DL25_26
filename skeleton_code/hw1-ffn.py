#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

import time
import utils


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, t, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """ Define a vanilla multiple-layer FFN with `layers` hidden layers 
        Args:
            n_classes (int)
            n_features (int)
            hidden_size (int)
            layers (int)
            activation_type (str)
            dropout (float): dropout probability
        """
        super().__init__()
        
        # Store parameters
        self.n_classes = t
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.layers = layers
        self.activation_type = activation_type
        self.dropout = dropout
        
        # Build the network layers
        layer_list = []
        
        # Input layer to first hidden layer
        layer_list.append(nn.Linear(n_features, hidden_size))
        
        # Activation function
        if activation_type == 'relu':
            layer_list.append(nn.ReLU())
        elif activation_type == 'tanh':
            layer_list.append(nn.Tanh())
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        
        # Dropout
        if dropout > 0:
            layer_list.append(nn.Dropout(dropout))
        
        # Additional hidden layers
        for _ in range(layers - 1):
            layer_list.append(nn.Linear(hidden_size, hidden_size))
            if activation_type == 'relu':
                layer_list.append(nn.ReLU())
            elif activation_type == 'tanh':
                layer_list.append(nn.Tanh())
            
            if dropout > 0:
                layer_list.append(nn.Dropout(dropout))
        
        # Output layer
        layer_list.append(nn.Linear(hidden_size, t))
        
        # Create the sequential model
        self.network = nn.Sequential(*layer_list)

    def forward(self, x, **kwargs):
        """ Compute a forward pass through the FFN
        Args:
            x (torch.Tensor): a batch of examples (batch_size x n_features)
        Returns:
            scores (torch.Tensor)
        """
        return self.network(x)
    
    
def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """ Do an update rule with the given minibatch
    Args:
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        model (nn.Module): a PyTorch defined model
        optimizer: optimizer used in gradient step
        criterion: loss function
    Returns:
        loss (float)
    """
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X)
    
    # Compute loss
    loss = criterion(outputs, y)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    return loss.item()


def predict(model, X):
    """ Predict the labels for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
    Returns:
        preds: (n_examples)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, preds = torch.max(outputs, 1)
    return preds


@torch.no_grad()
def evaluate(model, X, y, criterion):
    """ Compute the loss and the accuracy for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        criterion: loss function
    Returns:
        loss, accuracy (Tuple[float, float])
    """
    model.eval()
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Calculate accuracy
    _, preds = torch.max(outputs, 1)
    correct = (preds == y).sum().item()
    accuracy = correct / y.size(0)
    
    return loss.item(), accuracy


def plot(epochs, plottables, filename=None, ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=30, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=64, type=int,
                        help="Size of training batch.")
    parser.add_argument('-hidden_size', type=int, default=32)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-data_path', type=str, default='emnist-letters.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    train_X, train_y = dataset.X, dataset.y
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_feats = dataset.X.shape[1]

    print(f"N features: {n_feats}")
    print(f"N classes: {n_classes}")

    hidden_sizes = [32]
    learning_rates = [0.001]
    dropouts = [0]
    l2_decays = [0.0001]
    layers = [1, 3, 5, 7, 9]
    
    # Store all results
    all_results = []
    for opt.layers in layers:
        for hidden_size in hidden_sizes:
            for learning_rate in learning_rates:
                for dropout in dropouts:
                    for l2_decay in l2_decays:
                        print(f"\nTraining with - Hidden Size: {hidden_size}, Learning Rate: {learning_rate}, Dropout: {dropout}, L2 Decay: {l2_decay}")
                        # initialize the model
                        model = FeedforwardNetwork(
                            n_classes,
                            n_feats,
                            hidden_size,
                            opt.layers,
                            opt.activation,
                            dropout
                        )

                        # get an optimizer
                        optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

                        optim_cls = optims[opt.optimizer]
                        optimizer = optim_cls(
                            model.parameters(), lr=learning_rate, weight_decay=l2_decay
                        )

                        # get a loss criterion
                        criterion = nn.CrossEntropyLoss()

                        # training loop
                        epochs = torch.arange(0, opt.epochs + 1)  # Start from 0 to include initial evaluation
                        train_losses = []
                        train_accs = []
                        valid_losses = []
                        valid_accs = []

                        start = time.time()

                        model.eval()
                        initial_train_loss, initial_train_acc = evaluate(model, train_X, train_y, criterion)
                        initial_val_loss, initial_val_acc = evaluate(model, dev_X, dev_y, criterion)
                        train_losses.append(initial_train_loss)
                        train_accs.append(initial_train_acc)
                        valid_losses.append(initial_val_loss)
                        valid_accs.append(initial_val_acc)
                        print('initial val acc: {:.4f}'.format(initial_val_acc))

                        for ii in range(1, opt.epochs + 1):  # Start from 1 since epoch 0 is initial evaluation
                            print('Training epoch {}'.format(ii))
                            epoch_train_losses = []
                            model.train()
                            for X_batch, y_batch in train_dataloader:
                                loss = train_batch(
                                    X_batch, y_batch, model, optimizer, criterion)
                                epoch_train_losses.append(loss)

                            model.eval()
                            epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
                            _, train_acc = evaluate(model, train_X, train_y, criterion)
                            val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

                            print('train loss: {:.4f} | val loss: {:.4f} | val acc: {:.4f}'.format(
                                epoch_train_loss, val_loss, val_acc
                            ))

                            train_losses.append(epoch_train_loss)
                            train_accs.append(train_acc)
                            valid_losses.append(val_loss)
                            valid_accs.append(val_acc)

                        elapsed_time = time.time() - start
                        minutes = int(elapsed_time // 60)
                        seconds = int(elapsed_time % 60)
                        print('Training took {} minutes and {} seconds'.format(minutes, seconds))

                        _, test_acc = evaluate(model, test_X, test_y, criterion)
                        print('Final test acc: {:.4f}'.format(test_acc))

                        # Create unique config string for this specific configuration
                        config = (
                            f"batch-{opt.batch_size}-lr-{learning_rate}-epochs-{opt.epochs}-"
                            f"hidden-{hidden_size}-dropout-{dropout}-l2-{l2_decay}-"
                            f"layers-{opt.layers}-act-{opt.activation}-opt-{opt.optimizer}"
                        )

                        losses = {
                            "Train Loss": train_losses,
                            "Valid Loss": valid_losses,
                        }

                        plot(epochs, losses, filename=f'ffn-training-loss-{config}.pdf')
                        print(f"Final Training Accuracy: {train_accs[-1]:.4f}")
                        print(f"Best Validation Accuracy: {max(valid_accs):.4f}")
                        val_accuracy = { "Valid Accuracy": valid_accs }
                        plot(epochs, val_accuracy, filename=f'ffn-validation-accuracy-{config}.pdf')
                        
                        # Store results for this configuration
                        result = {
                            'hidden_size': hidden_size,
                            'learning_rate': learning_rate,
                            'dropout': dropout,
                            'l2_decay': l2_decay,
                            'final_train_acc': train_accs[-1],
                            'best_val_acc': max(valid_accs),
                            'final_test_acc': test_acc,
                            'val_accs': valid_accs.copy(),
                            'train_accs': train_accs.copy(),
                            'train_losses': train_losses.copy(),
                            'valid_losses': valid_losses.copy()
                        }
                        all_results.append(result)
    
    # Save all results summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL CONFIGURATIONS")
    print("="*80)
    print(f"{'Hidden':<8} {'LR':<8} {'Dropout':<8} {'L2':<8} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10}")
    print("-"*80)
    
    best_config = None
    best_val_acc = 0
    
    for result in all_results:
        print(f"{result['hidden_size']:<8} {result['learning_rate']:<8} {result['dropout']:<8} {result['l2_decay']:<8} "
              f"{result['final_train_acc']:<10.4f} {result['best_val_acc']:<10.4f} {result['final_test_acc']:<10.4f}")
        
        if result['best_val_acc'] > best_val_acc:
            best_val_acc = result['best_val_acc']
            best_config = result
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION:")
    print(f"Hidden Size: {best_config['hidden_size']}, Learning Rate: {best_config['learning_rate']}, "
          f"Dropout: {best_config['dropout']}, L2 Decay: {best_config['l2_decay']}")
    print(f"Best Validation Accuracy: {best_config['best_val_acc']:.4f}")
    print(f"Final Test Accuracy: {best_config['final_test_acc']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
