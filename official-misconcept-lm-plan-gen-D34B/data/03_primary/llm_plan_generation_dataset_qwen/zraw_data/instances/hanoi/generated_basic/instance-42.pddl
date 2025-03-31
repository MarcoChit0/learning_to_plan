


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d384 )
(:init
(smaller peg1 d384)
(smaller peg2 d384)
(smaller peg3 d384)
(clear peg2)
(clear peg3)
(clear d384)
(on d384 peg1)
)
(:goal
(and 
(on d384 peg3)
)
)
)


