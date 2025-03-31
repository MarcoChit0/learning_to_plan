


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d385 )
(:init
(smaller peg1 d385)
(smaller peg2 d385)
(smaller peg3 d385)
(clear peg2)
(clear peg3)
(clear d385)
(on d385 peg1)
)
(:goal
(and 
(on d385 peg3)
)
)
)


