


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d252 )
(:init
(smaller peg1 d252)
(smaller peg2 d252)
(smaller peg3 d252)
(clear peg2)
(clear peg3)
(clear d252)
(on d252 peg1)
)
(:goal
(and 
(on d252 peg3)
)
)
)


