


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d31 )
(:init
(smaller peg1 d31)
(smaller peg2 d31)
(smaller peg3 d31)
(clear peg2)
(clear peg3)
(clear d31)
(on d31 peg1)
)
(:goal
(and 
(on d31 peg3)
)
)
)


