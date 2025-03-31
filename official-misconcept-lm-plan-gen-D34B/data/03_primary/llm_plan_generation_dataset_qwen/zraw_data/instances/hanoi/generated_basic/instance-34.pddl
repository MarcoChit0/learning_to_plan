


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d127 )
(:init
(smaller peg1 d127)
(smaller peg2 d127)
(smaller peg3 d127)
(clear peg2)
(clear peg3)
(clear d127)
(on d127 peg1)
)
(:goal
(and 
(on d127 peg3)
)
)
)


