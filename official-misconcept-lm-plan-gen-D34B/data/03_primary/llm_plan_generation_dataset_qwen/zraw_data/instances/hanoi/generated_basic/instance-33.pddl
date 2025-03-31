


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d87 )
(:init
(smaller peg1 d87)
(smaller peg2 d87)
(smaller peg3 d87)
(clear peg2)
(clear peg3)
(clear d87)
(on d87 peg1)
)
(:goal
(and 
(on d87 peg3)
)
)
)


