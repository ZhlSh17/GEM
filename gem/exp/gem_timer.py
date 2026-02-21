import functools
from time import perf_counter
from collections.abc import Callable
from typing import Any

from gem.model import GEM


class GEM_Timer(GEM):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.latency = []
        self._in_generation = False

        self._model.forward = self.log_exec_time(self._model.forward)
        self._model.generate = self.disable_timer_in_generation(self._model.generate)

    def disable_timer_in_generation(self, func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._in_generation = True
            ret = func(*args, **kwargs)
            self._in_generation = False
            return ret

        return wrapper

    def log_exec_time(self, func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = perf_counter()
            ret = func(*args, **kwargs)
            end = perf_counter()
            if not self._in_generation:
                self.latency.append(end - start)

            return ret

        return wrapper
