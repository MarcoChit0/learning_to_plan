


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d228 )
(:init
(smaller peg1 d228)
(smaller peg2 d228)
(smaller peg3 d228)
(clear peg2)
(clear peg3)
(clear d228)
(on d228 peg1)
)
(:goal
(and 
(on d228 peg3)
)
)
)


