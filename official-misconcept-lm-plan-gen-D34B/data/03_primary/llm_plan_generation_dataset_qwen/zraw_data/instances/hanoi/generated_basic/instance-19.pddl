


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d468 )
(:init
(smaller peg1 d468)
(smaller peg2 d468)
(smaller peg3 d468)
(clear peg2)
(clear peg3)
(clear d468)
(on d468 peg1)
)
(:goal
(and 
(on d468 peg3)
)
)
)


