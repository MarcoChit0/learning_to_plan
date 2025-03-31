


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d395 d66 d98 )
(:init
(smaller peg1 d395)
(smaller peg1 d66)
(smaller peg1 d98)
(smaller peg2 d395)
(smaller peg2 d66)
(smaller peg2 d98)
(smaller peg3 d395)
(smaller peg3 d66)
(smaller peg3 d98)
(smaller d66 d395)
(smaller d98 d395)
(smaller d98 d66)
(clear peg2)
(clear peg3)
(clear d395)
(on d98 peg1)
(on d66 d98)
(on d395 d66)
)
(:goal
(and 
(on d98 peg3)
(on d66 d98)
(on d395 d66)
)
)
)


