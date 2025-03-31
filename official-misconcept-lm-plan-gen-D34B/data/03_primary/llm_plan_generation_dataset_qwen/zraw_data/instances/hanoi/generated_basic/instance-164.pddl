


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d49 d111 d421 d213 )
(:init
(smaller peg1 d49)
(smaller peg1 d111)
(smaller peg1 d421)
(smaller peg1 d213)
(smaller peg2 d49)
(smaller peg2 d111)
(smaller peg2 d421)
(smaller peg2 d213)
(smaller peg3 d49)
(smaller peg3 d111)
(smaller peg3 d421)
(smaller peg3 d213)
(smaller d111 d49)
(smaller d421 d49)
(smaller d213 d49)
(smaller d421 d111)
(smaller d213 d111)
(smaller d213 d421)
(clear peg2)
(clear peg3)
(clear d49)
(on d213 peg1)
(on d421 d213)
(on d111 d421)
(on d49 d111)
)
(:goal
(and 
(on d213 peg3)
(on d421 d213)
(on d111 d421)
(on d49 d111)
)
)
)


