With[{L = LogisticSigmoid},
    Manipulate[
        ParametricPlot[
            {Sin[Pi  (x + d)], JacobiSN[2  EllipticK[L[a]] x, L[a]]}, 
            {x, -1, 1}
        ], 
    {{d, 0}, -1, 1}, {a, -10, 10}]
]