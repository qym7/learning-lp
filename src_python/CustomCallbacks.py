import tensorflow as tf
import numpy as np

"""Module contains custom callbacks for training, prediction and evaluation of a tensorflow model."""


class EscapeLocalMinima(tf.keras.callbacks.LearningRateScheduler):
    """
    Custom learning rate scheduler that jumps if Adam is believed to be stuck in
    a non optimal local minimum.

    The precise jumping conditions are defined by the method stuck.

    Attributes
    ----------
    considered_epochs : int
        number of past epochs that will be taken into account for computing some
        jumping conditions.
    threshold : float
        highest acceptable loss value. Once the loss gets lower then threshold,
        the learning rate will not jump anymore.
    verbose : float
        if verbose > 0, learning rate is printed at the begin of every epoch
    """
    def __init__(self, scheduler, considered_epochs, threshold, verbose=0):
        super(EscapeLocalMinima, self).__init__(scheduler, verbose)
        self.considered_epochs = considered_epochs
        self.threshold = threshold
        self.errors = considered_epochs * [None]
        self.ind = 0
        self.last_jump = 0
        self.jump = False

    def clear_errors(self):
        for i in range(self.considered_epochs):
            self.errors[i] = None
        self.ind = 0

    def filled(self):
        if self.errors[-1] is not None:
            return True
        else:
            return False

    def mean(self):
        S = 0
        nb = 0
        for i in range(self.considered_epochs):
            elem = self.errors[i]
            if elem is not None:
                S += elem
                nb += 1
        return S / nb

    def stuck(self, error):
        ind = self.ind
        eind = (ind-1) % self.considered_epochs
        if error >= self.threshold and ((self.filled() and error >= self.mean())
                                        or (self.errors[eind] is not None and error >= 2 * self.errors[eind])):
            return True
        else:
            return False

    def on_epoch_begin(self, epoch, logs=None):
        if self.jump:
            self.last_jump = epoch
            self.jump = False
            self.clear_errors()
        super(EscapeLocalMinima, self).on_epoch_begin(epoch - self.last_jump, logs)

    def on_epoch_end(self, epoch, logs=None):
        super(EscapeLocalMinima, self).on_epoch_end(epoch, logs)
        error = logs["mean_absolute_percentage_error"]
        if self.stuck(error):
            self.jump = True
            print("jump")
        else:
            ind = self.ind
            self.errors[ind] = error
            self.ind = (ind + 1) % self.considered_epochs


