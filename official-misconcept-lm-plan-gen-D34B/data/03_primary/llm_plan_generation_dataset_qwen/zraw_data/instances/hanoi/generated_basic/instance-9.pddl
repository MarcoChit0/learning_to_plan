


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d218 )
(:init
(smaller peg1 d218)
(smaller peg2 d218)
(smaller peg3 d218)
(clear peg2)
(clear peg3)
(clear d218)
(on d218 peg1)
)
(:goal
(and 
(on d218 peg3)
)
)
)


