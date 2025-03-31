


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d73 )
(:init
(smaller peg1 d73)
(smaller peg2 d73)
(smaller peg3 d73)
(clear peg2)
(clear peg3)
(clear d73)
(on d73 peg1)
)
(:goal
(and 
(on d73 peg3)
)
)
)


