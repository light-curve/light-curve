import functools
import warnings


class ExperimentalWarning(UserWarning):
    pass


def warn_experimental(msg):
    warnings.warn(msg, category=ExperimentalWarning, stacklevel=2)


def mark_experimental(msg=None):
    if msg is None:
        msg = "Function {} is experimental and may cause any kind of troubles".format(f.__name__)

    def inner(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            warn_experimental(msg)
            return f(*args, **kwargs)

        return wrapped

    return inner


__all__ = (
    "ExperimentalWarning",
    "warn_experimental",
    "mark_experimental",
)
