import os
import numpy as np
import tensorflow as tf
from timeit import timeit

print('Tensorflow Version:',tf.__version__)

MODEL_DIR = './models'
MODEL_NAME = 'dualvae_2024-04-25'
N_RUNS = 100
BATCH_SIZE = 32

dvae = tf.keras.models.load_model(os.path.join(MODEL_DIR, MODEL_NAME))

inputData = np.zeros((BATCH_SIZE, 240, 1))

# Warmup run
dvae.X_encoder.predict(inputData)

modelRuntime = timeit(stmt='out = dvae.X_encoder.predict(inputData)', setup='from __main__ import dvae, inputData', number=N_RUNS)

print('Average encoder time for one batch: ',str(modelRuntime/N_RUNS),'(s)')