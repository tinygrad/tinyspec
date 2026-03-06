from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import weakref, hashlib
from uop import Ops

class UOpMetaClass(type):
  ucache:dict[tuple, weakref.ReferenceType[UOp]] = {}
  def __call__(cls, op:Ops, src:tuple[UOp,...]=tuple(), arg:Any=None, tag:Any=None):
    if (wret:=UOpMetaClass.ucache.get(key:=(op, src, arg, tag), None)) is not None and (ret:=wret()) is not None: return ret
    UOpMetaClass.ucache[key] = weakref.ref(created:=super().__call__(*key))
    return created

@dataclass(eq=False, slots=True, weakref_slot=True)
class UOp(metaclass=UOpMetaClass):
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
  def __repr__(self): return f"UOp({self.op.name}, arg={self.arg!r}, src={self.src})"
