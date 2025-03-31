


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d66 d85 )
(:init
(smaller peg1 d66)
(smaller peg1 d85)
(smaller peg2 d66)
(smaller peg2 d85)
(smaller peg3 d66)
(smaller peg3 d85)
(smaller d85 d66)
(clear peg2)
(clear peg3)
(clear d66)
(on d85 peg1)
(on d66 d85)
)
(:goal
(and 
(on d85 peg3)
(on d66 d85)
)
)
)


