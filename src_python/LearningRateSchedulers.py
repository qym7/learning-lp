import numpy as np

"""Module contains a list of learning rate schedulers for different models."""


def scheduler_opt_on_model_1_100_1(epoch):
    if epoch == 0:
        return 0.05
    elif epoch < 12:
        return 0.002 / (2 * epoch - 1)
    elif epoch < 20:
        return 0.00009
    else:
        return 0.00009 / (0.25 * epoch - 4)


def scheduler_LandS_opt_on_model_1_100_1(epoch):
    if epoch == 0:
        return 0.05
    elif epoch < 18:
        return 0.001
    else:
        return 0.001 / (epoch - 17)


def scheduler_LandS_opt_on_model_1_500_1(epoch):
    if epoch == 0:
        return 0.05
    elif epoch < 18:
        return 0.001
    elif epoch < 28:
        return 0.001 / (epoch - 17)
    else:
        return 0.0001 / (epoch - 27)


def scheduler_gbd_opt_on_model_1_500_1(epoch):
    if epoch == 0:
        return 0.05
    elif epoch < 6:
        return 0.001
    elif epoch < 18:
        return 0.001 / (epoch - 5)
    elif epoch < 46:
        return 0.00008 / ((epoch / 3) - 5)
    else:
        return 0.000008 / ((epoch / 3) - 15)


def scheduler_20term_opt_on_model_1_500_1(epoch):
    if epoch == 0:
        return 0.05
    elif epoch < 8:
        return 0.002 / (1 + (epoch - 1)/7)
    elif epoch < 18:
        return 0.001 / (epoch - 7)
    else:
        return 0.0001 / (epoch - 17)


def scheduler_20term_opt_on_model_input_2000_1000_1(epoch):
   if epoch == 0:
       return 0.05
   elif epoch < 6:
       return 0.005 / epoch
   elif epoch < 20:
       return 0.001
   else:
       return 0.001 / (1 + (epoch - 20)/4)


def scheduler_LandS_opt_on_model_input_5times40_1(epoch):
    if epoch == 0:
        return 0.05
    elif epoch <= 10:
        return 0.005
    elif epoch <= 25:
        return 0.001
    elif epoch <= 35:
        return 0.0001
    elif epoch <= 45:
        return 0.00001
    elif epoch <= 55:
        return 0.000001
    elif epoch <= 65:
        return 0.0000001
    else:
        return 0.00000001


def scheduler_ssn_opt_on_model_input_5times40_1(epoch):
    if epoch <= 20:
        return 0.01
    elif epoch <= 40:
        return 0.001
    elif epoch <= 60:
        return 0.0001
    else:
        return 0.000001

def schedulerI(epoch):
    if epoch < 5:
        return 0.1 / (epoch + 1)
    elif epoch < 10:
        return 0.01 / (2 * epoch - 9)
    elif epoch < 70:
        return 0.001 * np.exp(0.06 * (10 - epoch))
    else:
        return 0.001 * np.exp(0.06 * (10 - 70)) / (epoch - 69)


def schedulerII(epoch):
    return 0.1 / (10 * epoch + 1)


def schedulerIII(epoch):
    return 0.001
