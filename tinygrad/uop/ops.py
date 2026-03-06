from __future__ import annotations
from typing import Any, Callable
from dataclasses import dataclass
import weakref, hashlib, math
from tinygrad.uop import Ops, GroupOp
from tinygrad.dtype import DType, dtypes

# these are needed by mixin/movement.py -- must be defined before importing OpMixin
def resolve(x, default:bool=True):
  if isinstance(x, bool): return x
  return default
def smax(*lst):
  return max(*lst) if len(lst) > 1 else max(lst[0]) if hasattr(lst[0], '__iter__') else lst[0]

# import after resolve/smax are defined to avoid circular import
from tinygrad.mixin import OpMixin

# recursive_property replaces functools.cached_property in recursive UOp functions to prevent RecursionError
class recursive_property(property):
  def __init__(self, fxn):
    self.fxn = fxn
    self.nm = "_RECURSIVE_PROPERTY_"+fxn.__name__
    self.__doc__ = fxn.__doc__
  def __get__(self, x:UOp|None, owner=None):
    if x is None: return self
    for node in x.toposort(gate=lambda node: self.nm not in node.__dict__): node.__dict__[self.nm] = self.fxn(node)
    return x.__dict__[self.nm]

class UOpMetaClass(type):
  ucache:dict[tuple, weakref.ReferenceType[UOp]] = {}
  def __call__(cls, op:Ops, src:tuple[UOp,...]=tuple(), arg:Any=None, tag:Any=None):
    if (wret:=UOpMetaClass.ucache.get(key:=(op, src, arg, tag), None)) is not None and (ret:=wret()) is not None: return ret
    UOpMetaClass.ucache[key] = weakref.ref(created:=super().__call__(*key))
    return created

def _broadcast_shape(*shapes:tuple[int,...]) -> tuple[int,...]:
  max_dim = max(len(s) for s in shapes)
  aligned = [(1,)*(max_dim-len(s))+s for s in shapes]
  ret = []
  for dims in zip(*aligned):
    out = 1
    for d in dims:
      if d != 1:
        if out != 1 and out != d: raise ValueError(f"cannot broadcast shapes {shapes}")
        out = d
    ret.append(out)
  return tuple(ret)

@dataclass(eq=False, slots=True, weakref_slot=True)
class UOp(OpMixin, metaclass=UOpMetaClass):
  op:Ops
  src:tuple[UOp, ...] = tuple()
  arg:Any = None
  tag:Any = None
  def __del__(self):
    try: del UOpMetaClass.ucache[(self.op, self.src, self.arg, self.tag)]
    except (AttributeError, KeyError): pass
  def replace(self, **kwargs) -> UOp:
    new_args = (kwargs.pop("op", self.op), kwargs.pop("src", self.src), kwargs.pop("arg", self.arg), kwargs.pop("tag", self.tag))
    assert len(kwargs) == 0, f"unused kwargs in replace {list(kwargs)}"
    if (self.op, self.src, self.arg, self.tag) == new_args: return self
    return UOp(*new_args)
  @property
  def key(self) -> bytes:
    return hashlib.sha256(str((self.op, self.arg)).encode() + b"".join([s.key for s in self.src])).digest()
  def __repr__(self): return f"UOp({self.op.name}, arg={self.arg!r}, src={self.src})"

  def toposort(self, gate:Callable|None=None) -> dict[UOp, None]:
    cache: dict[UOp, None] = {}
    stack: list[tuple[UOp, bool]] = [(self, False)]
    while stack:
      node, visited = stack.pop()
      if node in cache: continue
      if not visited:
        if gate is None or gate(node):
          stack.append((node, True))
          for s in reversed(node.src): stack.append((s, False))
      else: cache[node] = None
    return cache

  # *** recursive properties (derived from spec) ***

  @recursive_property
  def dtype(self) -> DType:
    # source ops: dtype from arg
    if self.op in {Ops.BUFFER, Ops.BUFFER_VIEW, Ops.PARAM, Ops.CONST, Ops.VCONST}: return self.arg[1]
    # cast, bitcast: target dtype from arg
    if self.op in {Ops.CAST, Ops.BITCAST}: return self.arg
    # where: src[1].dtype
    if self.op is Ops.WHERE: return self.src[1].dtype
    # comparisons: bool
    if self.op in GroupOp.Comparison: return dtypes.bool
    # movement ops, reduce, ALU unary, ALU binary (non-cmp), copy, contiguous, detach, after: passthrough src[0]
    if self.op in GroupOp.Movement | GroupOp.Unary | (GroupOp.Binary - GroupOp.Comparison) | \
       {Ops.REDUCE, Ops.COPY, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD, Ops.DETACH, Ops.AFTER}:
      return self.src[0].dtype
    # call: body dtype
    if self.op is Ops.CALL: return self.src[0].dtype
    # range: index
    if self.op is Ops.RANGE: return dtypes.index
    # store, sink: void
    if self.op in {Ops.STORE, Ops.SINK}: return dtypes.void
    raise NotImplementedError(f"dtype not defined for {self.op}")

  @recursive_property
  def shape(self) -> tuple[sint, ...]:
    # source ops
    if self.op is Ops.BUFFER:
      size, dtype, device = self.arg
      return (len(device) * size,) if isinstance(device, tuple) else (size,)
    if self.op is Ops.BUFFER_VIEW: return (self.arg[0],)  # arg is (size, dtype, offset)
    if self.op is Ops.PARAM: return self.src[0].shape      # src[0] is shape UOp
    if self.op is Ops.CONST: return ()
    if self.op is Ops.VCONST: return (len(self.arg[0]),)   # arg is (values, dtype)

    # movement ops
    if self.op is Ops.PERMUTE: return tuple(self.src[0].shape[i] for i in self.arg)
    if self.op is Ops.FLIP: return self.src[0].shape
    if self.op is Ops.RESHAPE:
      new_shape = self.src[1].arg  # src[1] is the shape
      assert math.prod(self.src[0].shape) == math.prod(new_shape), f"reshape {self.src[0].shape} -> {new_shape}"
      return new_shape
    if self.op is Ops.EXPAND:
      new_shape = self.src[1].arg
      assert all(s == ns or s == 1 for s, ns in zip(self.src[0].shape, new_shape)), f"expand {self.src[0].shape} -> {new_shape}"
      return new_shape
    if self.op is Ops.PAD:
      b, e = self.src[1].arg, self.src[2].arg
      return tuple(s + b_k + e_k for s, b_k, e_k in zip(self.src[0].shape, b, e))
    if self.op is Ops.SHRINK:
      b, e = self.src[1].arg, self.src[2].arg
      return tuple(e_k - b_k for b_k, e_k in zip(b, e))
    if self.op is Ops.INDEX:
      ps, idx_srcs = self.src[0].shape, self.src[1:]
      result = []
      for idx in idx_srcs:
        if idx.shape == (): continue       # scalar index removes dim
        result.append(idx.shape[0])        # (k,)-shaped index
      result.extend(ps[len(idx_srcs):])    # remaining dims
      return tuple(result)
    if self.op is Ops.CAT:
      axis = self.arg
      if axis == -1: return (len(self.src),) + self.src[0].shape
      shapes = [s.shape for s in self.src]
      return tuple(sum(s[i] for s in shapes) if i == axis else shapes[0][i] for i in range(len(shapes[0])))

    # reduce: collapse axes to 1
    if self.op is Ops.REDUCE:
      reduce_op, axes = self.arg
      return tuple(1 if i in axes else s for i, s in enumerate(self.src[0].shape))

    # elementwise: broadcast all shaped inputs
    if self.op in GroupOp.ALU | {Ops.CAST, Ops.BITCAST, Ops.COPY, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD, Ops.DETACH}:
      shaped = [s.shape for s in self.src]
      if len(shaped) == 0: return ()
      if len(shaped) == 1: return shaped[0]
      return _broadcast_shape(*shaped)

    # call: body shape; range: scalar
    if self.op is Ops.CALL: return self.src[0].shape
    if self.op is Ops.RANGE: return ()
    # store/after: passthrough src[0]
    if self.op is Ops.STORE: return self.src[0].shape
    if self.op is Ops.AFTER: return self.src[0].shape
    raise NotImplementedError(f"shape not defined for {self.op}")

  # *** stubs for mixins ***

  def alu(self, op, *src): raise NotImplementedError
  def const_like(self, b): raise NotImplementedError
  def cast(self, dtype): raise NotImplementedError
  def _mop(self, op, arg): raise NotImplementedError

sint = int|UOp
