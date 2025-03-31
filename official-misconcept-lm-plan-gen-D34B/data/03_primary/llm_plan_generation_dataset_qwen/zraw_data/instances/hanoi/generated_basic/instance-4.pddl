


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d452 )
(:init
(smaller peg1 d452)
(smaller peg2 d452)
(smaller peg3 d452)
(clear peg2)
(clear peg3)
(clear d452)
(on d452 peg1)
)
(:goal
(and 
(on d452 peg3)
)
)
)


