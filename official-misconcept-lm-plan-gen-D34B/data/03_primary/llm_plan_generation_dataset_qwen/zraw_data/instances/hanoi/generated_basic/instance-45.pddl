


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d116 )
(:init
(smaller peg1 d116)
(smaller peg2 d116)
(smaller peg3 d116)
(clear peg2)
(clear peg3)
(clear d116)
(on d116 peg1)
)
(:goal
(and 
(on d116 peg3)
)
)
)


