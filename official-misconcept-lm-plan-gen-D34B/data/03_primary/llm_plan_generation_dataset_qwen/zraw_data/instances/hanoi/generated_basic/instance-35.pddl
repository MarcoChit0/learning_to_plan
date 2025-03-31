


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d488 )
(:init
(smaller peg1 d488)
(smaller peg2 d488)
(smaller peg3 d488)
(clear peg2)
(clear peg3)
(clear d488)
(on d488 peg1)
)
(:goal
(and 
(on d488 peg3)
)
)
)


