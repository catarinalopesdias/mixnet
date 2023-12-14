#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:37:08 2023

@author: catarinalopesdias
"""

import tensorflow as tf

############# AGC
def compute_norm(x, axis, keepdims):
    """
    Computes the euclidean norm of a tensor :math:`x`.

    Args:
        x: input tensor.
        axis: which axis to compute norm across.
        keepdims: whether to keep dimension after applying along axis.

    Returns:
        Euclidean norm.
    """
    return tf.math.reduce_sum(x**2, axis=axis, keepdims=keepdims) ** 0.5


def unitwise_norm(x):
    """
    Wrapper class which dynamically sets `axis` and `keepdims` given an
    input `x` for calculating euclidean norm.

    Args:
        x: input tensor.

    Returns:
        Euclidean norm.
    """
    if len(x.get_shape()) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
    elif len(x.get_shape()) in [
        2,
        3,
    ]:  # Linear layers of shape IO or multihead linear
        axis = 0
        keepdims = True
    elif len(x.get_shape()) == 4:  # Conv kernels of shape HWIO
        axis = [0, 1, 2]
        keepdims = True
    elif len(x.get_shape()) == 5:  # Conv kernels of shape HWDIO
        axis = [0, 1, 2, 3]
        keepdims = True
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 4, 5]! {x}")
    return compute_norm(x, axis, keepdims)


def adaptive_clip_grad(
    parameters, gradients, clip_factor: float = 0.01, eps: float = 1e-3
):
    """
    Performs adaptive gradient clipping on a given set of parameters and
    gradients.

    * Official JAX implementation (paper authors):
      https://github.com/deepmind/deepmind-research/tree/master/nfnets  # noqa
    * Ross Wightman's implementation
      https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py  # noqa

    Args:
        parameters: Which parameters to apply method on.
        gradients: Which gradients to apply clipping on.
        clip_factor: Sets upper limit for gradient clipping.
        eps: Epsilon - small number in :math:`max()` to avoid zero norm and
            preserve numerical stability.

    Returns:
        Updated gradients after gradient clipping.
    """
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        new_grads.append(new_grad)
    return new_grads



#################### GA###############
opt = tf.keras.optimizers.Optimizer
if int(tf.version.VERSION.split(".")[1]) > 10:
    opt = tf.keras.optimizers.legacy.Optimizer
    
#@tf.keras.utils.register_keras_serializable("gradient-accumulator")
    
class GradientAccumulateModel(tf.keras.Model):
    """Model wrapper for gradient accumulation."""

    def __init__(
        self,
        accum_steps: int = 1,
        mixed_precision: bool = False,
        use_agc: bool = False,
        clip_factor: float = 0.01,
        eps: float = 1e-3,
        experimental_distributed_support: bool = False,
        *args,
        **kwargs
    ):
        """Adds gradient accumulation support to existing Keras Model.

        Args:
            accum_steps: int > 0. Update gradient in every accumulation steps.
            mixed_precision: bool. Whether to enable mixed precision.
            use_agc: bool. Whether to enable adaptive gradient clipping.
            clip_factor: float > 0. Upper limit to gradient clipping.
            eps: float > 0. Small value to aid numerical stability.
            experimental_distributed_support: bool. Whether to enable
                experimental multi-gpu support. Only compatible with SGD. Can
                be used with other optimizers but we do not have complete
                control of the optimizer's state between accum_steps.
            **kwargs: keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.accum_steps = tf.constant(
            accum_steps, dtype=tf.int32, name="accum_steps"
        )
        self.accum_step_counter = tf.Variable(
            0,
            dtype=tf.int32,
            trainable=False,
            name="accum_counter",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        self.first_call = True
        self.mixed_precision = mixed_precision
        self.use_agc = use_agc
        self.clip_factor = clip_factor
        self.eps = eps
        self.experimental_distributed_support = experimental_distributed_support
        self.dtype_value = self.dtype
        self.gradient_accumulation = None
        self.reinit_grad_accum()

    def train_step(self, data):
        """Performs single train step."""
        # need to reinit accumulator for models subclassed from tf.keras.Model
        if self.first_call:
            self.reinit_grad_accum()
            self.first_call = False

        self.accum_step_counter.assign_add(1)

        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        # NOTE that x and y are lists of inputs and outputs,
        # hence this wrapper supports multi-input-output models
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # forward pass

            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )
            loss = loss / tf.cast(
                self.accum_steps, loss.dtype
            )  # MEAN reduction here IMPORTANT! Don't use SUM!

            # scale loss if mixed precision is enabled
            if self.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        # Calculate batch gradients -> these are scaled gradients if mixed
        # precision is enabled
        gradients = tape.gradient(
            loss,
            self.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

        # scale gradients if mixed precision is enabled
        if self.mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        # apply adaptive gradient clipping -> should be AFTER unscaling
        if self.use_agc:
            gradients = adaptive_clip_grad(
                self.trainable_variables,
                gradients,
                clip_factor=self.clip_factor,
                eps=self.eps,
            )

        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(
                gradients[i], read_value=False
            )

        # accumulate gradients only after certain number of steps
        # self.accum_steps.assign(self.accum_steps * tf.cast(tf.logical_not(\
        #   tf.equal(self.accum_step_counter,self.accum_steps)), tf.int32))
        if not self.experimental_distributed_support:
            tf.cond(
                tf.equal(self.accum_step_counter, self.accum_steps),
                true_fn=self.apply_accu_gradients,
                false_fn=lambda: None,
            )

        else:
            # NOTE: This enabled multi-gpu support, but only for SGD (!)
            should_apply = tf.equal(self.accum_step_counter, self.accum_steps)
            logical_grads = [
                tf.cast(should_apply, grad_component.dtype) * grad_component
                for grad_component in self.gradient_accumulation
            ]
            self.optimizer.apply_gradients(
                zip(logical_grads, self.trainable_variables)
            )
            self.accum_step_counter.assign(
                self.accum_step_counter
                * tf.cast(tf.logical_not(should_apply), tf.int32)
            )
            for i in range(len(self.gradient_accumulation)):
                self.gradient_accumulation[i].assign_add(-1 * logical_grads[i])

        # update metrics
        self.compiled_metrics.update_state(
            y, y_pred, sample_weight=sample_weight
        )
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        """Performs gradient update and resets slots afterwards."""
        # apply accumulated gradients
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.trainable_variables)
        )

        # reset
        self.accum_step_counter.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(
                    self.trainable_variables[i], dtype=self.dtype_value
                ),
                read_value=False,
            )

    def reinit_grad_accum(self):
        """Reinitialized gradient accumulator slots."""
        # reinitialize gradient accumulator
        self.gradient_accumulation = [
            tf.Variable(
                tf.zeros_like(v, dtype=self.dtype_value),
                trainable=False,
                name="accum_" + str(i),
                synchronization=tf.VariableSynchronization.ON_READ,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )
            for i, v in enumerate(self.trainable_variables)
        ]


#