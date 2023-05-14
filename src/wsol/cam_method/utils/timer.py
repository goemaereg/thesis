import torch
from torchvision.models import mobilenet_v2
import time, gc
import numpy as np


class Timer:
    _gc_disable_count = 0
    timers = {}

    @classmethod
    def reset(cls):
        _gc_disable_count = 0
        cls.timers = {}

    @classmethod
    def _gc_disable(cls):
        if cls._gc_disable_count == 0:
            gc.disable()
        cls._gc_disable_count += 1

    @classmethod
    def _gc_enable(cls):
        cls._gc_disable_count -= 1
        if cls._gc_disable_count < 0:
            raise ValueError
        if cls._gc_disable_count == 0:
            gc.enable()

    @classmethod
    def exists(cls, name):
        return name in cls.timers

    @classmethod
    def create_or_get_timer(cls, device, name, warm_up=False):
        if not cls.exists(name):
            cls.timers[name] = Timer(device, name, warm_up)
        return cls.timers[name]

    def __init__(self, device, name, warm_up=False):
        self.device = device
        self.name = name
        self.values = []
        self.unit = 1e-3
        if 'cuda' in device:
            self.unit = 1e-3
            self.elapsed_ms = 0.0
            self.total_elapsed_ms = 0.0
            if warm_up:
                self.warm_up()
        else:
            self.unit = 1e-9
            self.elapsed_ns = 0
            self.total_elapsed_ns = 0.0

    def get_total_elapsed_ms(self):
        if 'cuda' in self.device:
            return self.total_elapsed_ms
        else:
            return self.total_elapsed_ns / 1e6

    def get_sum(self):
        return np.sum(np.asarray(self.values))

    def get_mean_std(self):
        values = np.asarray(self.values)
        mean = np.mean(values)
        std = np.std(values)
        return mean, std

    def warm_up(self):
        if 'cuda' not in self.device:
            return
        model = mobilenet_v2(weights=None)
        model.to(self.device)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(self.device)
            for _ in range(10):
                _ = model(dummy_input)
        torch.cuda.synchronize() # wait for warm up to complete actions on GPU

    def start(self):
        self._gc_disable()
        if 'cuda' in self.device:
            torch.cuda.synchronize()
            self.starter = torch.cuda.Event(enable_timing=True)
            self.starter.record()
        else:
            self.counter_start = time.monotonic_ns()
    def stop(self):
        # wait for gpu sync
        if 'cuda' in self.device:
            self.ender = torch.cuda.Event(enable_timing=True)
            self.ender.record()
            self.ender.wait()
            self.ender.synchronize()
            torch.cuda.synchronize()
            elapsed = self.starter.elapsed_time(self.ender)
            self.elapsed_ms = self.starter.elapsed_time(self.ender)
            self.total_elapsed_ms += self.elapsed_ms
        else:
            self.counter_stop = time.monotonic_ns()
            self.elapsed_ns = self.counter_stop - self.counter_start
            elapsed = self.counter_stop - self.counter_start
            self.total_elapsed_ns += self.elapsed_ns
        self._gc_enable()
        self.values.append(elapsed)

    def __str__(self):
        return f'Timer({self.__dict__}))'
