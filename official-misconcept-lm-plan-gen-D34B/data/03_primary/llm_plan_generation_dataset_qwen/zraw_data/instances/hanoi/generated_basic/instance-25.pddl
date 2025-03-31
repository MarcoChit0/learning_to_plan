


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d429 )
(:init
(smaller peg1 d429)
(smaller peg2 d429)
(smaller peg3 d429)
(clear peg2)
(clear peg3)
(clear d429)
(on d429 peg1)
)
(:goal
(and 
(on d429 peg3)
)
)
)


