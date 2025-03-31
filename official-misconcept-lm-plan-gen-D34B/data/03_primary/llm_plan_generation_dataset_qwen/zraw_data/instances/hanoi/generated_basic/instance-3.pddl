


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d328 )
(:init
(smaller peg1 d328)
(smaller peg2 d328)
(smaller peg3 d328)
(clear peg2)
(clear peg3)
(clear d328)
(on d328 peg1)
)
(:goal
(and 
(on d328 peg3)
)
)
)


