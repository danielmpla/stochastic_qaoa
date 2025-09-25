from sympy import symbols, pprint, simplify, expand, Symbol

class Binary(Symbol):
    def __new__(cls, name):
        obj = Symbol.__new__(cls, name, integer=True, nonnegative=True, bounded=True)
        return obj

    @property
    def constraints(self):
        return self >= 0, self <= 1

    def _eval_power(self, other):
        return self

js = symbols([f"j_{i}" for i in range(2)], cls=Binary)
sells = symbols([f"sell_{i}" for i in range(2)], cls=Binary)
buys = symbols([f"buy_{i}" for i in range(2)], cls=Binary)
ps = symbols([f"p_{i}" for i in range(2)], cls=Binary)

j = 2**1 * js[0] + 2**0 * js[1]
sell = 2**1 * sells[0] + 2**0 * sells[1]
buy = 2**1 * buys[0] + 2**0 * buys[1]
pv = 2 ** 1 * ps[0] + 2 ** 0 * ps[1]

P = 5

z = - j * 0.25 + buy * 0.4 - 0.1 * sell + expand(5 * (j - buy + sell - pv) ** 2) + P * buy * sell

pprint(simplify(z))
