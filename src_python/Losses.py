from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.keras.losses import Loss
import numpy as np

"""Module implements different loss functions."""


class MeanLogarithmicError(Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        quotient = math_ops.divide(y_pred, y_true)
        quotient = math_ops.exp(10 - 10 * math_ops.sign(quotient)) * math_ops.abs(quotient) + 0.000000001
        return 100 * K.mean(math_ops.abs(math_ops.log(quotient)), axis=-1)


class SymmetricMeanAbsolutePercentageError(Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return 200 * K.mean(math_ops.divide(math_ops.abs(math_ops.subtract(y_pred, y_true)),
                      math_ops.add(math_ops.abs(y_pred), math_ops.abs(y_true))), axis=-1)
