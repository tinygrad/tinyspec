from __future__ import annotations
from tinygrad.uop.ops import UOp, sint
from tinygrad.dtype import DType

def _dispatch(self, name, *args, **kwargs):
  args = tuple(a.uop if isinstance(a, Tensor) else a for a in args)
  ret = getattr(self.uop, name)(*args, **kwargs)
  return Tensor(ret) if isinstance(ret, UOp) else ret

class Tensor:
  def __init__(self, data):
    if isinstance(data, UOp): self.uop = data
    elif isinstance(data, (list, tuple)): self.uop = UOp.from_list(data)
    else: raise TypeError(f"can't create Tensor from {type(data)}")
  def __repr__(self): return f"<Tensor {self.shape} {self.dtype}>"
  @property
  def dtype(self) -> DType: return self.uop.dtype
  @property
  def shape(self) -> tuple[sint, ...]: return self.uop.shape

  def __getattr__(self, name):
    attr = getattr(self.uop, name)
    if not callable(attr): return attr
    return lambda *args, **kwargs: _dispatch(self, name, *args, **kwargs)

# forward dunders that UOp's mixins define (not inherited from object)
# _override_dunders: object defines these but UOp's mixins redefine them, so we must forward
_object_dunders = set(dir(object))
_override_dunders = {"__lt__", "__le__", "__gt__", "__ge__", "__ne__", "__eq__"}
for _name in dir(UOp):
  if _name.startswith("__") and _name.endswith("__") and (_name not in _object_dunders or _name in _override_dunders) \
     and callable(getattr(UOp, _name)):
    setattr(Tensor, _name, lambda self, *a, _n=_name, **kw: _dispatch(self, _n, *a, **kw))
