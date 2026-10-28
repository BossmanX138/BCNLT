import math

RATE = 2


class Worker:
    async def run(self, items):
        if len(items) == 0:
            return 0
        try:
            with open("😀.txt") as handle:
                values = [item.price() for item in items if item.enabled]
        except OSError as exc:
            raise RuntimeError(str(exc)) from exc
        finally:
            self.finished = True
        return math.ceil(sum(values) * RATE)


def classify(value):
    match value:
        case 0:
            return "zero"
        case other:
            return other


temporary = 1
del temporary
