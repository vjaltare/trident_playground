"""
TridentMOE Layer.
Defines a Mixture of Experts Layer with a set of "experts" and a "routing" matrix for the experts to communicate.
Can be sequentially arranged to form a deep network.

TODO: Testing this script
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx.nn import initializers
import math

from utils import trident

class TridentMOELayer(nnx.Module):

    def __init__(self, 
                 in_features: int, # input feature dimension
                 num_experts: int, # number of smaller experts
                 expert_size: int, # size of each expert
                 rngs: nnx.Rngs,
                 thresholds: list = [-0.5, 0.5], # thresholds for ternary activation
                 noise_sd: float = 0.1, # standard deviation of the noise injected before ternary activation
                 ):
        
        self.in_features = in_features
        self.num_experts = num_experts
        self.expert_size = expert_size
        self.rngs = rngs
        self.thresholds = thresholds
        self.noise_sd = noise_sd
        
        # compute the size of the routing matrix
        in_chunks = math.ceil(in_features/expert_size) # find the number of chuks of in the input that the experts will handle

        glorot_init = initializers.glorot_normal()
        self.routing_matrix = nnx.Param(
            glorot_init(rngs.params(), (in_chunks, num_experts))
        )

        # define the weights of the experts
        self.We = nnx.Param(
            glorot_init(rngs.params(), (num_experts, expert_size, expert_size))
        )

    def __call__(self, x: jnp.array) -> jnp.array:
        """
        Forward pass through the layer
        Args:
        x: input array of shape (samples/batch size, in_features)
        """

        # reshape x into chunks of expert size
        x = x.reshape(x.shape[0], -1, self.expert_size)

        # print(x.shape)
        # print(self.routing_matrix.shape)

        # make sure that the the routing matrix is ternary
        ternary_routing = trident(self.routing_matrix.value, self.thresholds, self.noise_sd, self.rngs.dropout())

        # route the inputs to the experts
        x_routed = jnp.einsum('ik, bij -> bkj', ternary_routing, x) # shape (batch size, num_experts, expert_size)

        # pass the routed inputs through experts
        x_experts = jnp.einsum('eij, bej -> bei', self.We.value, x_routed) # shape (batch size, num_experts, expert_size)

        # combine the outputs of the experts
        y = x_experts.reshape(x.shape[0], -1) # shape (batch size, num_experts * expert_size)

        return y
    
    def get_routing_matrix(self, apply_activation: bool = False):
        if apply_activation:
            return trident(self.routing_matrix.value, self.thresholds, self.noise_sd, self.rngs.dropout())
        else:
            return self.routing_matrix.value




