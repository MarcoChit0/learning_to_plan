


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d229 )
(:init
(smaller peg1 d229)
(smaller peg2 d229)
(smaller peg3 d229)
(clear peg2)
(clear peg3)
(clear d229)
(on d229 peg1)
)
(:goal
(and 
(on d229 peg3)
)
)
)


