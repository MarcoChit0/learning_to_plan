


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d73 d252 )
(:init
(smaller peg1 d73)
(smaller peg1 d252)
(smaller peg2 d73)
(smaller peg2 d252)
(smaller peg3 d73)
(smaller peg3 d252)
(smaller d252 d73)
(clear peg2)
(clear peg3)
(clear d73)
(on d252 peg1)
(on d73 d252)
)
(:goal
(and 
(on d252 peg3)
(on d73 d252)
)
)
)


