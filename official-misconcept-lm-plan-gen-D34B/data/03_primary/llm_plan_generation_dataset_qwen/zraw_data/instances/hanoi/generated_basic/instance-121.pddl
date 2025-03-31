


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d173 d117 d399 )
(:init
(smaller peg1 d173)
(smaller peg1 d117)
(smaller peg1 d399)
(smaller peg2 d173)
(smaller peg2 d117)
(smaller peg2 d399)
(smaller peg3 d173)
(smaller peg3 d117)
(smaller peg3 d399)
(smaller d117 d173)
(smaller d399 d173)
(smaller d399 d117)
(clear peg2)
(clear peg3)
(clear d173)
(on d399 peg1)
(on d117 d399)
(on d173 d117)
)
(:goal
(and 
(on d399 peg3)
(on d117 d399)
(on d173 d117)
)
)
)


