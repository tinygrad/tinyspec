from __future__ import annotations
import functools, operator, os, math
from typing import TypeVar, Iterable, Sequence, Any, TypeGuard

T = TypeVar("T")

def prod(x:Iterable[T]) -> T|int: return functools.reduce(operator.mul, x, 1)
def dedup(x:Iterable[T]) -> list[T]: return list(dict.fromkeys(x))
def argfix(*x):
  if x and x[0].__class__ in (tuple, list):
    if len(x) != 1: raise ValueError(f"bad arg {x}")
    return tuple(x[0])
  return x
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__))
def all_same(items:Sequence) -> bool: return all(x == items[0] for x in items)
def all_int(t:Sequence[Any]) -> TypeGuard[tuple[int, ...]]: return all(isinstance(s, int) for s in t)
def flatten(l:Iterable[Iterable[T]]) -> list[T]: return [item for sublist in l for item in sublist]
def make_tuple(x:int|Sequence[int], cnt:int) -> tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else tuple(x)
def ceildiv(num, amt):
  if hasattr(num, 'vmin') and num.vmin >= 0 and (amt > 0 if isinstance(amt, int) else amt.vmin > 0): return (num + amt - 1) // amt
  return int(ret) if isinstance((ret:=-(num//-amt)), float) else ret
def polyN(x:T, p:list[float]) -> T: return functools.reduce(lambda acc,c: acc*x+c, p, 0.0)  # type: ignore
@functools.cache
def getenv(key:str, default:Any=0): return type(default)(os.getenv(key, default))

import contextlib
from typing import ClassVar, Generic

class Context(contextlib.ContextDecorator):
  def __init__(self, **kwargs): self.kwargs = kwargs
  def __enter__(self):
    self.old_context:dict[str, Any] = {k: ContextVar._cache[k].value for k in self.kwargs}
    for k,v in self.kwargs.items(): ContextVar._cache[k].value = v
  def __exit__(self, *args):
    for k,v in self.old_context.items(): ContextVar._cache[k].value = v

class ContextVar(Generic[T]):
  _cache: ClassVar[dict[str, ContextVar]] = {}
  value: T
  key: str
  def __init__(self, key: str, default_value: T):
    if key in ContextVar._cache: raise RuntimeError(f"attempt to recreate ContextVar {key}")
    ContextVar._cache[key] = self
    self.value, self.key = getenv(key, default_value), key
  def __bool__(self): return bool(self.value)
  def __eq__(self, x): return self.value == x
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  def __lt__(self, x): return self.value < x

DEBUG = ContextVar("DEBUG", 0)
