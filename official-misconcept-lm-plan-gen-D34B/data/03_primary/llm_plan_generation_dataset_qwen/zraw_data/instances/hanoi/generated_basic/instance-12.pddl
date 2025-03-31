


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d7 )
(:init
(smaller peg1 d7)
(smaller peg2 d7)
(smaller peg3 d7)
(clear peg2)
(clear peg3)
(clear d7)
(on d7 peg1)
)
(:goal
(and 
(on d7 peg3)
)
)
)


