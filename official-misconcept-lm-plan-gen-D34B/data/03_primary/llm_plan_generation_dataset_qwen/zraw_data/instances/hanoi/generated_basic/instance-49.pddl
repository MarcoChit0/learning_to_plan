


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d418 )
(:init
(smaller peg1 d418)
(smaller peg2 d418)
(smaller peg3 d418)
(clear peg2)
(clear peg3)
(clear d418)
(on d418 peg1)
)
(:goal
(and 
(on d418 peg3)
)
)
)


