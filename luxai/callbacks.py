"""
Custom callbacks
"""
import time
import os
import gc
import psutil
import warnings
import socket
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, TensorBoard
import nvidia_smi


class LogLearningRate(Callback):
    """
    Simple callback that logs the learning rate. It is intended to be use along CSVLogger.
    When using both callbacks the evolution of the learning rate will be saved to csv file.

    Example ::

        LogLearningRate(),
        CSVLogger(os.path.join(model_folder, 'training.log'), append=True),

    """
    # pylint: disable=W0613

    def on_epoch_end(self, epoch, logs=None):
        logs['lr'] = K.eval(self.model.optimizer.lr)


class LogEpochTime(Callback):
    """
    Simple callback that logs the epoch time. It is intended to be use along CSVLogger.

    Example ::

        LogEpochTime(),
        CSVLogger(os.path.join(model_folder, 'training.log'), append=True),

    """
    # pylint: disable=W0613

    def __init__(self):
        super(LogEpochTime, self).__init__()
        self._start_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs['epoch_time'] = time.time() - self._start_time


class LogETA(Callback):
    """
    Simple callback that logs the ETA (Estimate Time Arrival).
    It is intended to be use along CSVLogger.

    Example ::

        LogETA(max_epochs),
        CSVLogger(os.path.join(model_folder, 'training.log'), append=True),

    """
    # pylint: disable=W0613

    def __init__(self, max_epochs):
        super().__init__()
        self._epoch_time = None
        self._start_time = 0
        self._max_epochs = max_epochs

    def on_epoch_begin(self, epoch, logs=None):
        self._start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        current_epoch_time = time.time() - self._start_time
        if self._epoch_time is None:
            self._epoch_time = current_epoch_time
        else:
            alpha = 0.5
            self._epoch_time = alpha*current_epoch_time + (1-alpha)*self._epoch_time
        eta = (self._max_epochs - epoch - 1)*self._epoch_time
        logs['ETA_seconds'] = eta
        logs['ETA_hours'] = eta/3600
        logs['ETA_days'] = eta/3600/24


def get_ram_usage_and_available():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss/1e9
    stats = psutil.virtual_memory()  # returns a named tuple
    ram_available = getattr(stats, 'available')/1e9
    return ram_usage, ram_available


def get_cpu_usage():
    return psutil.cpu_percent()


class LogRAM(Callback):
    """
    Simple callback that logs the RAM. It is intended to be use along CSVLogger.
    When using both callbacks the evolution of the learning rate will be saved to csv file.

    Example ::

        LogLearningRate(),
        CSVLogger(os.path.join(model_folder, 'training.log'), append=True),

    """
    # pylint: disable=W0613
    @staticmethod
    def on_epoch_end(epoch, logs=None):
        ram_usage, ram_available = get_ram_usage_and_available()
        logs['ram_usage_GB'] = ram_usage
        logs['ram_available_GB'] = ram_available


class LogCPU(Callback):
    """
    Simple callback that logs the CPU. It is intended to be use along CSVLogger.
    When using both callbacks the evolution of the learning rate will be saved to csv file.

    Example ::

        LogLearningRate(),
        CSVLogger(os.path.join(model_folder, 'training.log'), append=True),

    """
    # pylint: disable=W0613

    def __init__(self):
        super(LogCPU, self).__init__()
        self._cpu_usage = []

    def on_epoch_start(self, epoch, logs=None):
        self._cpu_usage = []

    def on_epoch_end(self, epoch, logs=None):
        if self._cpu_usage:
            logs['cpu_usage'] = np.mean(self._cpu_usage)

    def on_batch_end(self, batch, logs=None):
        self._cpu_usage.append(get_cpu_usage())


class LogGPU(Callback):
    """
    Simple callback that logs the GPU. It is intended to be use along CSVLogger.
    When using both callbacks the evolution of the learning rate will be saved to csv file.

    Example ::

        LogLearningRate(),
        CSVLogger(os.path.join(model_folder, 'training.log'), append=True),

    """
    # pylint: disable=W0613

    def __init__(self):
        super(LogGPU, self).__init__()
        self._gpu_handles = {}
        try:
            nvidia_smi.nvmlInit()
            for gpu_idx in get_available_gpu_index():
                self._gpu_handles[gpu_idx] = nvidia_smi.nvmlDeviceGetHandleByIndex(
                    gpu_idx)
        except nvidia_smi.NVMLError as exc:
            print(exc)

        self._gpu_usage = {}
        self._gpu_memory = {}
        self._gpu_temperature = {}
        self.n_gpus = len(self._gpu_handles)

    def on_epoch_start(self, epoch, logs=None):
        self._gpu_usage = {}
        self._gpu_memory = {}
        self._gpu_temperature = {}

    def on_epoch_end(self, epoch, logs=None):
        if self._gpu_usage:
            for gpu_idx, values in self._gpu_usage.items():
                logs['gpu%i_usage' % gpu_idx] = np.mean(values)
        if self._gpu_memory:
            for gpu_idx, values in self._gpu_memory.items():
                logs['gpu%i_memory' % gpu_idx] = np.mean(values)
        if self._gpu_temperature:
            for gpu_idx, values in self._gpu_temperature.items():
                logs['gpu%i_temp' % gpu_idx] = np.mean(values)

    def on_batch_end(self, batch, logs=None):
        """ Save gpu stats on each batch """
        for gpu_idx, gpu_handle in self._gpu_handles.items():
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(gpu_handle)
            if gpu_idx not in self._gpu_usage:
                self._gpu_usage[gpu_idx] = []
            if gpu_idx not in self._gpu_memory:
                self._gpu_memory[gpu_idx] = []
            self._gpu_usage[gpu_idx].append(res.gpu)
            self._gpu_memory[gpu_idx].append(res.memory)

            if gpu_idx not in self._gpu_temperature:
                self._gpu_temperature[gpu_idx] = []
            temperature = nvidia_smi.nvmlDeviceGetTemperature(gpu_handle, 0)
            self._gpu_temperature[gpu_idx].append(temperature)


class GarbageCollector(Callback):
    """
    Simple callback that collects garbage at the end of each epoch
    """
    # pylint: disable=W0613, R0201

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


def get_available_gpu_index():
    """ Returns a list with the available gpu index """
    # pylint: disable= R1705
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        if os.environ['CUDA_VISIBLE_DEVICES']:
            return [int(idx) for idx in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        else:
            return []
    else:
        return list(range(nvidia_smi.nvmlDeviceGetCount()))


class LogConstantValue(Callback):
    """
    """
    def __init__(self, key, value):
        super().__init__()
        self.key = key
        self.value = value
        self.already_logged = False

    def on_epoch_end(self, epoch, logs=None):
        if not self.already_logged:
            logs[self.key] = self.value
            self.already_logged = True

