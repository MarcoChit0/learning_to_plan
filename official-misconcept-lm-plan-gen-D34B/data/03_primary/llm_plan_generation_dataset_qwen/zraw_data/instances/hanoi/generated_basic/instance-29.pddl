


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d275 )
(:init
(smaller peg1 d275)
(smaller peg2 d275)
(smaller peg3 d275)
(clear peg2)
(clear peg3)
(clear d275)
(on d275 peg1)
)
(:goal
(and 
(on d275 peg3)
)
)
)


