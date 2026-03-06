# flake8: noqa: E702
from enum import auto, IntEnum

class Ops(IntEnum):
  # source ops
  BUFFER = auto(); BUFFER_VIEW = auto()
  PARAM = auto()
  CONST = auto(); VCONST = auto()

  # movement ops
  PERMUTE = auto(); FLIP = auto(); RESHAPE = auto(); EXPAND = auto()
  PAD = auto(); SHRINK = auto(); INDEX = auto(); CAT = auto()

  # reduce ops
  REDUCE = auto()

  # elementwise unary
  EXP2 = auto(); LOG2 = auto(); SIN = auto(); SQRT = auto(); NEG = auto(); RECIP = auto(); TRUNC = auto()
  CAST = auto(); BITCAST = auto(); COPY = auto()
  CONTIGUOUS = auto(); CONTIGUOUS_BACKWARD = auto(); DETACH = auto()

  # elementwise binary
  ADD = auto(); MUL = auto(); MAX = auto(); MOD = auto(); IDIV = auto()
  CMPLT = auto(); CMPNE = auto()
  XOR = auto(); OR = auto(); AND = auto(); SHR = auto(); SHL = auto()
  THREEFRY = auto()

  # elementwise ternary
  WHERE = auto()

  # call ops
  CALL = auto(); RANGE = auto()

  # store ops
  STORE = auto()

  # ordering ops
  AFTER = auto(); SINK = auto()

  # program ops
  PROGRAM = auto(); LINEAR = auto(); INS = auto(); SOURCE = auto(); BINARY = auto()

  # reduce op arguments
  NOOP = auto()

class GroupOp:
  Unary = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.NEG, Ops.RECIP, Ops.TRUNC}
  Binary = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.MOD, Ops.IDIV, Ops.CMPLT, Ops.CMPNE,
            Ops.XOR, Ops.OR, Ops.AND, Ops.SHR, Ops.SHL, Ops.THREEFRY}
  Ternary = {Ops.WHERE}
  ALU = set.union(Unary, Binary, Ternary)
  Elementwise = set.union(ALU, {Ops.CAST, Ops.BITCAST})
  Movement = {Ops.PERMUTE, Ops.FLIP, Ops.RESHAPE, Ops.EXPAND, Ops.PAD, Ops.SHRINK, Ops.INDEX, Ops.CAT}
  Commutative = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.XOR, Ops.AND, Ops.OR}
  Associative = {Ops.ADD, Ops.MUL, Ops.AND, Ops.OR, Ops.MAX}
  Idempotent = {Ops.OR, Ops.AND, Ops.MAX}
  Comparison = {Ops.CMPLT, Ops.CMPNE}
