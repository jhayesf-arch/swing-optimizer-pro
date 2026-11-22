import pickle, io
ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype'),
           ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', '_reconstruct'),
           ('numpy._core.multiarray', 'scalar'), ('builtins', 'dict'), ('builtins', 'list'),
           ('collections', 'OrderedDict'), ('numpy._core.numeric', '_frombuffer'), ('numpy.core.numeric', '_frombuffer')}
class Safe(pickle.Unpickler):
    def find_class(self, mod, name):
        if (mod, name) in ALLOWED:
            return super().find_class(mod, name)
        raise pickle.UnpicklingError(f'blocked {mod}.{name}')
def load(path):
    return Safe(open(path, 'rb')).load()
