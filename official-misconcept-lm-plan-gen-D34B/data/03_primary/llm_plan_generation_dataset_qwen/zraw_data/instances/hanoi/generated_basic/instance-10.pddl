


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d36 )
(:init
(smaller peg1 d36)
(smaller peg2 d36)
(smaller peg3 d36)
(clear peg2)
(clear peg3)
(clear d36)
(on d36 peg1)
)
(:goal
(and 
(on d36 peg3)
)
)
)


