


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d456 )
(:init
(smaller peg1 d456)
(smaller peg2 d456)
(smaller peg3 d456)
(clear peg2)
(clear peg3)
(clear d456)
(on d456 peg1)
)
(:goal
(and 
(on d456 peg3)
)
)
)


