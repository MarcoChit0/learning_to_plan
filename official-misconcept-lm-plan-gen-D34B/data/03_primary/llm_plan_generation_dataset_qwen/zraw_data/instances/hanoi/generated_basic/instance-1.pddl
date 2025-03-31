


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d196 )
(:init
(smaller peg1 d196)
(smaller peg2 d196)
(smaller peg3 d196)
(clear peg2)
(clear peg3)
(clear d196)
(on d196 peg1)
)
(:goal
(and 
(on d196 peg3)
)
)
)


