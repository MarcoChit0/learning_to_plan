


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d145 d440 d44 )
(:init
(smaller peg1 d145)
(smaller peg1 d440)
(smaller peg1 d44)
(smaller peg2 d145)
(smaller peg2 d440)
(smaller peg2 d44)
(smaller peg3 d145)
(smaller peg3 d440)
(smaller peg3 d44)
(smaller d440 d145)
(smaller d44 d145)
(smaller d44 d440)
(clear peg2)
(clear peg3)
(clear d145)
(on d44 peg1)
(on d440 d44)
(on d145 d440)
)
(:goal
(and 
(on d44 peg3)
(on d440 d44)
(on d145 d440)
)
)
)


