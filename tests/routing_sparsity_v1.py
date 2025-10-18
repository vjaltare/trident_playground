"""
How does routing sparsity affect performance of a MOE model?
- Trident framework provides a way to construct ternary routing matrices.
- As a first step, we check how the network performs on a set of random test inputs and labels with varying levels of routing sparsity.

Author: Vikrant Jaltare
Date Created: 10/17/2025
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx.nn import initializers
import optax
from typing import Callable
from tqdm import tqdm
from datetime import date


import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from functools import partial
from collections import defaultdict
import os
import pickle

from models import TridentMOELayer
from utils import trident

DATA_PATH = "/Volumes/export/isn/vikrant/Data/trident/playground/data"
today = date.today().isoformat()

# ------------------------------------------------------
# Save data
# ------------------------------------------------------
def save_metrics(metrics_dict, filename, metrics_dir=DATA_PATH):
    """
    Save the metrics to a file.
    Args:
        metrics_dict: dict, metrics to save.
        filename: str, name of the file to save the metrics to.
    """

    # metrics_dir = "/local_disk/vikrant/scrramble/logs"
    filename = os.path.join(metrics_dir, filename)

    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure the directory exists.

    with open(filename, 'wb') as f:
        pickle.dump(metrics_dict, f)

    print(f"Metrics saved to {filename}")


# -----------------------------------------------------
# Define the model
# -----------------------------------------------------
class TridentMOENet(nnx.Module):

    def __init__(self, 
                 in_features: int, 
                 expert_size: int, 
                 rngs: nnx.Rngs,
                 thresholds: list = [-0.5, 0.5], 
                 noise_sd: float = 0.1, 
                 layer_sizes: nnx.Data[list] = nnx.List([30, 10]), # in terms of number of experts
                 ):
        self.in_features = in_features
        self.expert_size = expert_size
        self.rngs = rngs
        self.thresholds = thresholds
        self.noise_sd = noise_sd
        self.in_chunks = math.ceil(in_features/expert_size)

        layer_sizes.insert(0, self.in_chunks)
        self.layer_sizes = layer_sizes

        self.layers = nnx.List([ # CAUTION: new flax needs this nnx.List!!!
            TridentMOELayer(
                in_features=li*self.expert_size,
                num_experts=lo,
                expert_size=self.expert_size,
                rngs=self.rngs,
                thresholds=self.thresholds,
                noise_sd=self.noise_sd
            )

            for li, lo in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ])

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x

# -----------------------------------------------------
# Test Data Loader
# -----------------------------------------------------
def generate_random_inputs_and_labels(
        batch_size: int,
        in_features:int,
        label_size: int,
        rngs: nnx.Rngs,
):
    test_inputs = jax.random.normal(rngs.params(), (batch_size, in_features))
    test_labels = jax.random.randint(rngs.params(), (batch_size, ), 0, label_size)
    return test_inputs, test_labels

# -----------------------------------------------------
# Training Functions
# -----------------------------------------------------
# ce loss function
def loss_function(model: TridentMOENet,
                  inputs: jax.Array,
                  labels: jax.Array,
                  num_categories: int = 10):

    
    logits = model(inputs) # forward pass

    # reshape the inputs
    logits = logits.reshape(logits.shape[0], num_categories, -1)

    # find the mean across the last dimension
    logits = jnp.mean(logits, axis=-1) # shape of logits is (batch size, num_categories)

    # compute cross entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    return loss, logits

# weighted loss function
# def weighted_loss_function(
#         model: TridentMOENet,
#         inputs: jax.Array,
#         labels: jax.Array,
        
# ):


# training step
@nnx.jit
def train_step(model: TridentMOENet,
               optimizer: nnx.Optimizer,
               metrics: nnx.MultiMetric,
               inputs: jax.Array,
               labels: jax.Array,
               loss_fn: Callable = loss_function,
               ):
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, inputs, labels)
    metrics.update(loss=loss, logits=logits, labels=labels)
    optimizer.update(model, grads) # new changes

# evaluation function
@nnx.jit
def eval_step(
    model: TridentMOENet,
    metrics: nnx.MultiMetric,
    inputs: jax.Array,
    labels: jax.Array,
    loss_fn: Callable = loss_function,
):
    loss, logits = loss_fn(model, inputs, labels)
    metrics.update(loss=loss, logits=logits, labels=labels)

# prediction step
@nnx.jit
def pred_step(
    model: TridentMOENet,
    inputs: jax.Array
):
    logits = model(inputs) # forward pass

    # reshape the inputs
    logits = logits.reshape(logits.shape[0], -1, 10)

    # find the mean across the last dimension
    logits = jnp.mean(logits, axis=-1) # shape of logits is (batch size, num_categories)

    # prediction is the argmax
    preds = logits.argmax(axis=-1)

    return preds

# -----------------------------------------------------
# Pipeline
# -----------------------------------------------------
def sparsity_analysis(
    threshold: float, # [-threshold, threshold] will be used as the thresholds for trident
    inputs: jax.Array,
    labels: jax.Array,
    rngs: nnx.Rngs,
    num_epochs: int = 1000,
    eval_every: int = 50,
):
    # define the model
    model = TridentMOENet(
        in_features=inputs.shape[1],
        expert_size=10,
        rngs=rngs,
        thresholds=[-threshold, threshold],
        noise_sd=0.1
    )

    # optimizers
    hyperparameters = {
        'learning_rate': 1e-3, # 1e-3 seems to work well
        'momentum': 0.9, 
        'weight_decay': 1e-4
    }

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay']),
        wrt=nnx.Param
    )

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss')
    )

    metrics_history = defaultdict(list)
    results_dict = defaultdict(list)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        train_step(model, optimizer, metrics, inputs, labels)

        if epoch > 0 and (epoch % eval_every == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch+1}/{num_epochs}")

            # for metric, value in metrics.compute().items():
            #     metrics_history[f"train_{metric}"].append(float(value))
            # metrics.reset()

            eval_step(model, metrics, inputs, labels)

            for metric, value in metrics.compute().items():
                metrics_history[f"eval_{metric}"].append(float(value))
            metrics.reset()       

        # print(metrics_history)

            print(f"Epoch {epoch+1}, Loss: {metrics_history['eval_loss'][-1]:.4f}, Accuracy: {metrics_history['eval_accuracy'][-1]*100:.2f}%")

    # # append the performance and threshold to results_dict
    # results_dict['threshold'].append(float(threshold))
    # results_dict['accuracy'].append(float(metrics_history['eval_accuracy'][-1]))
    # results_dict['loss'].append(float(metrics_history['eval_loss'][-1]))
    # results_dict['resample'].append(int(resample))

    return model, metrics_history








# testing
def __main__():
    thresholds_arr = jnp.logspace(-4, 0, 20).tolist()
    key1 = jax.random.key(234)
    key1, key2 = jax.random.split(key1, 2)
    rngs = nnx.Rngs(params=key1, dropout=key2)
    num_resamples = 30
    results_dict = defaultdict(list)
    test_input, test_labels = generate_random_inputs_and_labels(batch_size=10, in_features=100, label_size=10, rngs=rngs)
    for r in tqdm(range(num_resamples), desc="Resample Progress"):
        for i_th, th in tqdm(enumerate(thresholds_arr), total=len(thresholds_arr), desc="Sparsity Analysis Progress"):
            print(f"Testing for threshold {th}")
            key1, key2 = jax.random.split(key1, 2)
            rngs = nnx.Rngs(params=key1, dropout=key2)

            model, metrics_history = sparsity_analysis(
                threshold=th,
                inputs=test_input,
                labels=test_labels,
                rngs=rngs,
            )

            # append the performance and threshold to results_dict
            results_dict['threshold'].append(float(th))
            results_dict['accuracy'].append(float(metrics_history['eval_accuracy'][-1]))
            results_dict['loss'].append(float(metrics_history['eval_loss'][-1]))
            results_dict['resample'].append(int(r))

    # print(results_dict)

    # save the results_dict
    save_metrics(results_dict, filename=f"routing_sparsity_n{model.noise_sd:.02f}_r{num_resamples}_{today}.pkl", metrics_dir=DATA_PATH)

if __name__ == "__main__":
    __main__()

