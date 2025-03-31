


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d437 )
(:init
(smaller peg1 d437)
(smaller peg2 d437)
(smaller peg3 d437)
(clear peg2)
(clear peg3)
(clear d437)
(on d437 peg1)
)
(:goal
(and 
(on d437 peg3)
)
)
)


