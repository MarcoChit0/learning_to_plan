


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d177 )
(:init
(smaller peg1 d177)
(smaller peg2 d177)
(smaller peg3 d177)
(clear peg2)
(clear peg3)
(clear d177)
(on d177 peg1)
)
(:goal
(and 
(on d177 peg3)
)
)
)


