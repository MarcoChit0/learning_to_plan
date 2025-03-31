


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d111 )
(:init
(smaller peg1 d111)
(smaller peg2 d111)
(smaller peg3 d111)
(clear peg2)
(clear peg3)
(clear d111)
(on d111 peg1)
)
(:goal
(and 
(on d111 peg3)
)
)
)


