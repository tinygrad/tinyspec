from __future__ import annotations
from typing import Final
import functools
from dataclasses import dataclass
from tinygrad.helpers import getenv

ConstType = float|int|bool
PyConst = float|int|bool

class DTypeMetaClass(type):
  dcache: dict[tuple, DType] = {}
  def __call__(cls, *args, **kwargs):
    if (ret:=DTypeMetaClass.dcache.get(args, None)) is not None: return ret
    DTypeMetaClass.dcache[args] = ret = super().__call__(*args)
    return ret

@dataclass(frozen=True, eq=False)
class DType(metaclass=DTypeMetaClass):
  priority: int
  bitsize: int
  name: str
  fmt: str|None
  count: int
  _scalar: DType|None
  @property
  def itemsize(self) -> int: return (self.bitsize + 7) // 8
  @staticmethod
  def new(priority:int, bitsize:int, name:str, fmt:str|None): return DType(priority, bitsize, name, fmt, 1, None)
  def __repr__(self): return f"dtypes.{INVERSE_DTYPES_DICT[self.scalar().name]}"+(f".vec({self.count})" if self.count != 1 else "")
  def __lt__(self, o:DType): return (self.priority, self.bitsize, self.name, self.fmt, self.count) < (o.priority, o.bitsize, o.name, o.fmt, o.count)
  @functools.cache
  def vec(self, sz:int) -> DType:
    assert self.count == 1, f"can't vectorize {self} with size {sz}"
    if sz == 1 or self == dtypes.void: return self
    return DType(self.priority, self.bitsize*sz, f"{INVERSE_DTYPES_DICT[self.name]}{sz}", None, sz, self)
  def scalar(self) -> DType: return self._scalar if self._scalar is not None else self
  @property
  def base(self): return self

class dtypes:
  @staticmethod
  @functools.cache
  def is_float(x: DType) -> bool: return x.scalar() in dtypes.floats
  @staticmethod
  @functools.cache
  def is_int(x: DType) -> bool: return x.scalar() in (dtypes.ints + (dtypes.index,))
  @staticmethod
  @functools.cache
  def is_unsigned(x: DType) -> bool: return x.scalar() in dtypes.uints
  @staticmethod
  def is_bool(x: DType) -> bool: return x.scalar() == dtypes.bool
  @staticmethod
  @functools.cache
  def min(dtype:DType):
    if dtypes.is_int(dtype): return 0 if dtypes.is_unsigned(dtype) else -2**(dtype.scalar().bitsize-1)
    return -float("inf") if dtypes.is_float(dtype) else False
  @staticmethod
  @functools.cache
  def max(dtype:DType):
    if dtypes.is_int(dtype): return 2**(dtype.scalar().bitsize)-1+dtypes.min(dtype)
    return float("inf") if dtypes.is_float(dtype) else True
  void: Final[DType] = DType.new(-1, 0, "void", None)
  index: Final[DType] = DType.new(-1, 800, "index", None)
  bool: Final[DType] = DType.new(0, 1, "bool", '?')
  int8: Final[DType] = DType.new(1, 8, "signed char", 'b')
  uint8: Final[DType] = DType.new(2, 8, "unsigned char", 'B')
  int16: Final[DType] = DType.new(3, 16, "short", 'h')
  uint16: Final[DType] = DType.new(4, 16, "unsigned short", 'H')
  int32: Final[DType] = DType.new(5, 32, "int", 'i')
  uint32: Final[DType] = DType.new(6, 32, "unsigned int", 'I')
  int64: Final[DType] = DType.new(7, 64, "long", 'q')
  uint64: Final[DType] = DType.new(8, 64, "unsigned long", 'Q')
  float16: Final[DType] = DType.new(11, 16, "half", 'e')
  bfloat16: Final[DType] = DType.new(12, 16, "__bf16", None)
  float32: Final[DType] = DType.new(13, 32, "float", 'f')
  float64: Final[DType] = DType.new(14, 64, "double", 'd')

  # dtype aliases
  half = float16; float = float32; double = float64 # noqa: E702
  uchar = uint8; ushort = uint16; uint = uint32; ulong = uint64 # noqa: E702
  char = int8; short = int16; int = int32; long = int64 # noqa: E702

  default_float = float32
  default_int = int32

  floats = (float16, bfloat16, float32, float64)
  uints = (uint8, uint16, uint32, uint64)
  sints = (int8, int16, int32, int64)
  ints = uints + sints

DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if isinstance(v, DType) and not k.startswith(("default", "void", "index", "_"))}
INVERSE_DTYPES_DICT = {**{v.name:k for k,v in DTYPES_DICT.items()}, "void": "void", "index":"index"}

# https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html
promo_lattice = { dtypes.bool: [dtypes.int8, dtypes.uint8], dtypes.int8: [dtypes.int16], dtypes.int16: [dtypes.int32],
  dtypes.int32: [dtypes.int64], dtypes.int64: [dtypes.uint64], dtypes.uint8: [dtypes.int16, dtypes.uint16],
  dtypes.uint16: [dtypes.int32, dtypes.uint32], dtypes.uint32: [dtypes.int64, dtypes.uint64],
  dtypes.uint64: [dtypes.float16, dtypes.bfloat16],
  dtypes.float16: [dtypes.float32], dtypes.bfloat16: [dtypes.float32], dtypes.float32: [dtypes.float64], }

@functools.cache
def _get_recursive_parents(dtype:DType) -> set[DType]:
  return set.union(*[_get_recursive_parents(d) for d in promo_lattice[dtype]], {dtype}) if dtype != dtypes.float64 else {dtypes.float64}
@functools.cache
def least_upper_dtype(*ds:DType) -> DType:
  return min(set.intersection(*[_get_recursive_parents(d.scalar()) for d in ds]))
def least_upper_float(dt:DType) -> DType: return dt if dtypes.is_float(dt) else least_upper_dtype(dt, dtypes.default_float)
