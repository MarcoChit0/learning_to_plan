


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d57 d223 d262 )
(:init
(smaller peg1 d57)
(smaller peg1 d223)
(smaller peg1 d262)
(smaller peg2 d57)
(smaller peg2 d223)
(smaller peg2 d262)
(smaller peg3 d57)
(smaller peg3 d223)
(smaller peg3 d262)
(smaller d223 d57)
(smaller d262 d57)
(smaller d262 d223)
(clear peg2)
(clear peg3)
(clear d57)
(on d262 peg1)
(on d223 d262)
(on d57 d223)
)
(:goal
(and 
(on d262 peg3)
(on d223 d262)
(on d57 d223)
)
)
)


