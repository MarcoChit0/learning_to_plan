


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d25 d116 )
(:init
(smaller peg1 d25)
(smaller peg1 d116)
(smaller peg2 d25)
(smaller peg2 d116)
(smaller peg3 d25)
(smaller peg3 d116)
(smaller d116 d25)
(clear peg2)
(clear peg3)
(clear d25)
(on d116 peg1)
(on d25 d116)
)
(:goal
(and 
(on d116 peg3)
(on d25 d116)
)
)
)


