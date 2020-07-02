from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.keras.losses import Loss

"""Module implements different loss functions."""


class RelativeLogarithmicError(Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.mean(math_ops.abs(math_ops.log(math_ops.abs(math_ops.divide(y_pred, y_true))+0.00000001)), axis=-1)
