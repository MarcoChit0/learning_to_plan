


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d72 )
(:init
(smaller peg1 d72)
(smaller peg2 d72)
(smaller peg3 d72)
(clear peg2)
(clear peg3)
(clear d72)
(on d72 peg1)
)
(:goal
(and 
(on d72 peg3)
)
)
)


