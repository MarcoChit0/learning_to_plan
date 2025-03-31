


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d477 )
(:init
(smaller peg1 d477)
(smaller peg2 d477)
(smaller peg3 d477)
(clear peg2)
(clear peg3)
(clear d477)
(on d477 peg1)
)
(:goal
(and 
(on d477 peg3)
)
)
)


