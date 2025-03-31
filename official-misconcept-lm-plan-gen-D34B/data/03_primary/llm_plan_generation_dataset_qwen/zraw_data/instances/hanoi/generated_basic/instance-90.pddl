


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d74 d261 )
(:init
(smaller peg1 d74)
(smaller peg1 d261)
(smaller peg2 d74)
(smaller peg2 d261)
(smaller peg3 d74)
(smaller peg3 d261)
(smaller d261 d74)
(clear peg2)
(clear peg3)
(clear d74)
(on d261 peg1)
(on d74 d261)
)
(:goal
(and 
(on d261 peg3)
(on d74 d261)
)
)
)


