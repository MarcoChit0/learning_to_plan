


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d252 d35 )
(:init
(smaller peg1 d252)
(smaller peg1 d35)
(smaller peg2 d252)
(smaller peg2 d35)
(smaller peg3 d252)
(smaller peg3 d35)
(smaller d35 d252)
(clear peg2)
(clear peg3)
(clear d252)
(on d35 peg1)
(on d252 d35)
)
(:goal
(and 
(on d35 peg3)
(on d252 d35)
)
)
)


