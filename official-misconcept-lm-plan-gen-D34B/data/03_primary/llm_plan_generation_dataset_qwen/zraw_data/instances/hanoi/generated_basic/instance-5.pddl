


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d105 )
(:init
(smaller peg1 d105)
(smaller peg2 d105)
(smaller peg3 d105)
(clear peg2)
(clear peg3)
(clear d105)
(on d105 peg1)
)
(:goal
(and 
(on d105 peg3)
)
)
)


