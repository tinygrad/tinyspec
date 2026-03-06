import unittest
from tinygrad.tensor import Tensor

class TestTiny(unittest.TestCase):
  def test_plus(self):
    out = Tensor([1.,2,3]) + Tensor([4.,5,6])
    print(out.uop)

if __name__ == '__main__':
  unittest.main()
