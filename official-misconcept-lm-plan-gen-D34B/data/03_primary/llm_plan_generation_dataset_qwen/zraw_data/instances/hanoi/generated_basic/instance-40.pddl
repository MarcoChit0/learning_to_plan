


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d310 )
(:init
(smaller peg1 d310)
(smaller peg2 d310)
(smaller peg3 d310)
(clear peg2)
(clear peg3)
(clear d310)
(on d310 peg1)
)
(:goal
(and 
(on d310 peg3)
)
)
)


