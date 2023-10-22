from typing import Generator, Generic, Sequence, TypeVar

_T = TypeVar("_T")


def divup(a: int, b: int) -> int:
    return (a + b - 1) // b


class RunlengthBlock(tuple, Generic[_T]):
    val: _T
    begin: int
    len: int

    def __new__(cls, val: _T, begin: int, len: int):
        self = tuple.__new__(cls, (val, begin, len))
        self.val = val
        self.begin = begin
        self.len = len
        return self

    @property
    def end(self) -> int:
        return self.begin + self.len


def runlength(xs: Sequence[_T]) -> Generator[RunlengthBlock[_T], None, None]:
    if len(xs) == 0:
        return

    last, begin = xs[0], 0
    for i, x in enumerate(xs):
        if x != last:
            yield RunlengthBlock(val=last, begin=begin, len=i - begin)
            last, begin = x, i

    yield RunlengthBlock(val=last, begin=begin, len=len(xs) - begin)
