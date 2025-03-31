


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d171 )
(:init
(smaller peg1 d171)
(smaller peg2 d171)
(smaller peg3 d171)
(clear peg2)
(clear peg3)
(clear d171)
(on d171 peg1)
)
(:goal
(and 
(on d171 peg3)
)
)
)


