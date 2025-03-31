


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d398 d112 d277 )
(:init
(smaller peg1 d398)
(smaller peg1 d112)
(smaller peg1 d277)
(smaller peg2 d398)
(smaller peg2 d112)
(smaller peg2 d277)
(smaller peg3 d398)
(smaller peg3 d112)
(smaller peg3 d277)
(smaller d112 d398)
(smaller d277 d398)
(smaller d277 d112)
(clear peg2)
(clear peg3)
(clear d398)
(on d277 peg1)
(on d112 d277)
(on d398 d112)
)
(:goal
(and 
(on d277 peg3)
(on d112 d277)
(on d398 d112)
)
)
)


