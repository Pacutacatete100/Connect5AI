import tensorflow as tf
import numpy as np
import os
import random as rd
from PIL import  Image
from pyscreenshot import grab
import matplotlib.pyplot as plt
##import Agent

discount_rate = 0.95


def discount_normalize_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    total_rewards = 0

    for i in reversed(range(len(rewards))):
        total_rewards = total_rewards * discount_rate + rewards[i]
        discounted_rewards[i] = total_rewards

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    return discounted_rewards