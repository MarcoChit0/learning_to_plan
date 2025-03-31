


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d173 )
(:init
(smaller peg1 d173)
(smaller peg2 d173)
(smaller peg3 d173)
(clear peg2)
(clear peg3)
(clear d173)
(on d173 peg1)
)
(:goal
(and 
(on d173 peg3)
)
)
)


