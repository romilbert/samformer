# coding=utf-8
# MIT License
# 
# Copyright (c) 2024 Romain Ilbert
# 
# Based on the non-official PyTorch implementation of Sharpness-Aware Minimization (SAM) found at:
# https://github.com/davda54/sam, which is an implementation of the paper:
# "Sharpness-Aware Minimization for Efficiently Improving Generalization"
# by Foret, Kleiner, Mobahi, and Neyshabur. The original PyTorch implementation is also under the MIT License.
# 
# This TensorFlow implementation of SAM is for our specific needs and aims to provide similar functionality
# as the original PyTorch version, adhering to the principles of enhancing model performance while minimizing
# loss sharpness for improved generalization.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Implementation of Sharpness Aware Minimization."""

import tensorflow as tf


class SAM:
    """
    Sharpness-Aware Minimization (SAM) for Enhanced Training Stability.
    
    SAM optimizes a model's parameters in the direction that enhances model
    performance while simultaneously minimizing loss sharpness, aiming to improve
    generalization. This implementation wraps around a base TensorFlow optimizer
    to apply the SAM methodology.
    
    Reference:
    "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    by Foret, Kleiner, Mobahi, and Neyshabur. https://openreview.net/pdf?id=6Tm1mposlrM

    The original implementation can be found at:
    - https://github.com/davda54/sam
    
    Attributes:
        base_optimizer (tf.keras.optimizers.Optimizer): The TensorFlow optimizer to wrap.
        rho (float): The neighborhood size for sharpness-aware optimization.
        eps (float): A small epsilon value to prevent division by zero.
    """

    def __init__(self, base_optimizer, rho=0.05, eps=1e-12):
        """
        Initializes the SAM optimizer wrapper.
        
        Parameters:
            base_optimizer (tf.keras.optimizers.Optimizer): The base optimizer.
            rho (float): The neighborhood size for sharpness-aware optimization.
            eps (float): A small epsilon value to prevent division by zero.
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.rho = rho
        self.eps = eps
        self.base_optimizer = base_optimizer

    def first_step(self, gradients, trainable_vars):
        """
        Performs the first optimization step, moving weights in the direction
        that increases loss sharpness.
        
        Parameters:
            gradients (List[tf.Tensor]): Gradients of the loss with respect to the model parameters.
            trainable_vars (List[tf.Variable]): The model's trainable variables.
        """
        self.e_ws = []
        grad_norm = tf.linalg.global_norm(gradients)
        ew_multiplier = self.rho / (grad_norm + self.eps)

        for i in range(len(trainable_vars)):
            e_w = tf.math.multiply(gradients[i], ew_multiplier)
            trainable_vars[i].assign_add(e_w)
            self.e_ws.append(e_w)

    def second_step(self, gradients, trainable_variables):
        """
        Performs the second optimization step, applying the base optimizer
        update after reverting the first step's perturbation.
        
        Parameters:
            gradients (List[tf.Tensor]): Gradients of the loss with respect to the model parameters after the first step.
            trainable_variables (List[tf.Variable]): The model's trainable variables.
        """
        for i in range(len(trainable_variables)):
            trainable_variables[i].assign_add(-self.e_ws[i])  # Revert first step
        self.base_optimizer.apply_gradients(zip(gradients, trainable_variables))
