"""Training code for the v3 truss-game model.

Imported by the Colab notebook rather than copied into it. The v1 notebook held
its own copy of the game classes, they drifted, and the deployed model became
impossible to reproduce -- the code that actually trained it was never in the
repository. One definition, in here.
"""

from . import data, metrics

__all__ = ['data', 'metrics']
