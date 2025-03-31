


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d84 d499 d325 )
(:init
(smaller peg1 d84)
(smaller peg1 d499)
(smaller peg1 d325)
(smaller peg2 d84)
(smaller peg2 d499)
(smaller peg2 d325)
(smaller peg3 d84)
(smaller peg3 d499)
(smaller peg3 d325)
(smaller d499 d84)
(smaller d325 d84)
(smaller d325 d499)
(clear peg2)
(clear peg3)
(clear d84)
(on d325 peg1)
(on d499 d325)
(on d84 d499)
)
(:goal
(and 
(on d325 peg3)
(on d499 d325)
(on d84 d499)
)
)
)


