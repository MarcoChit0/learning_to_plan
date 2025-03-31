


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d491 d283 d132 )
(:init
(smaller peg1 d491)
(smaller peg1 d283)
(smaller peg1 d132)
(smaller peg2 d491)
(smaller peg2 d283)
(smaller peg2 d132)
(smaller peg3 d491)
(smaller peg3 d283)
(smaller peg3 d132)
(smaller d283 d491)
(smaller d132 d491)
(smaller d132 d283)
(clear peg2)
(clear peg3)
(clear d491)
(on d132 peg1)
(on d283 d132)
(on d491 d283)
)
(:goal
(and 
(on d132 peg3)
(on d283 d132)
(on d491 d283)
)
)
)


