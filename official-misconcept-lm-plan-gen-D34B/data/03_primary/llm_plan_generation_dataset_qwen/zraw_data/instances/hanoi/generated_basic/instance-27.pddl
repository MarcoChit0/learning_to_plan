


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d295 )
(:init
(smaller peg1 d295)
(smaller peg2 d295)
(smaller peg3 d295)
(clear peg2)
(clear peg3)
(clear d295)
(on d295 peg1)
)
(:goal
(and 
(on d295 peg3)
)
)
)


