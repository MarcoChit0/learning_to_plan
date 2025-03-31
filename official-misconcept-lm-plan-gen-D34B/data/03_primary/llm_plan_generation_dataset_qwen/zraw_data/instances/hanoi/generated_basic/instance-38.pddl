


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d245 )
(:init
(smaller peg1 d245)
(smaller peg2 d245)
(smaller peg3 d245)
(clear peg2)
(clear peg3)
(clear d245)
(on d245 peg1)
)
(:goal
(and 
(on d245 peg3)
)
)
)


