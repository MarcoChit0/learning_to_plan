


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d489 )
(:init
(smaller peg1 d489)
(smaller peg2 d489)
(smaller peg3 d489)
(clear peg2)
(clear peg3)
(clear d489)
(on d489 peg1)
)
(:goal
(and 
(on d489 peg3)
)
)
)


