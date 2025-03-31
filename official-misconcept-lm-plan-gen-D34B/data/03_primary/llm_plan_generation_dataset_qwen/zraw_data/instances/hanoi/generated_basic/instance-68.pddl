


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d69 d491 )
(:init
(smaller peg1 d69)
(smaller peg1 d491)
(smaller peg2 d69)
(smaller peg2 d491)
(smaller peg3 d69)
(smaller peg3 d491)
(smaller d491 d69)
(clear peg2)
(clear peg3)
(clear d69)
(on d491 peg1)
(on d69 d491)
)
(:goal
(and 
(on d491 peg3)
(on d69 d491)
)
)
)


