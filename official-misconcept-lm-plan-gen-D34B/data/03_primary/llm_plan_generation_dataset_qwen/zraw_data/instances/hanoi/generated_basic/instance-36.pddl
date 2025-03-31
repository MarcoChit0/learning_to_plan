


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d287 )
(:init
(smaller peg1 d287)
(smaller peg2 d287)
(smaller peg3 d287)
(clear peg2)
(clear peg3)
(clear d287)
(on d287 peg1)
)
(:goal
(and 
(on d287 peg3)
)
)
)


