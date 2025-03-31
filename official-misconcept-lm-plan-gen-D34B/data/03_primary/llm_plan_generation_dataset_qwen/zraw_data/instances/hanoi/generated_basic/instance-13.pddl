


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d377 )
(:init
(smaller peg1 d377)
(smaller peg2 d377)
(smaller peg3 d377)
(clear peg2)
(clear peg3)
(clear d377)
(on d377 peg1)
)
(:goal
(and 
(on d377 peg3)
)
)
)


