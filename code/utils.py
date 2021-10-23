#!/usr/env/bin python
"""
utility functions to use for the scripts
"""

import os


def wrap_savefig(f):
    """
    decorator for plt.savefig
    """

    def wrap(*args, **kwargs):
        path = os.path.join("..", "figures")
        if "fname" in kwargs.keys():
            kwargs["fname"] = os.path.join(path, kwargs["fname"])
            out = f(*args, **kwargs, bbox_inches="tight")
        else:
            out = f(
                os.path.join(path, args[0]), *args[1:], **kwargs, bbox_inches="tight"
            )
        return out

    return wrap
