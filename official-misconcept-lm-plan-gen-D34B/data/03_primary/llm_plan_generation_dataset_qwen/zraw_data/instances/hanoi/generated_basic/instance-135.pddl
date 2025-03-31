


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d87 d33 d21 )
(:init
(smaller peg1 d87)
(smaller peg1 d33)
(smaller peg1 d21)
(smaller peg2 d87)
(smaller peg2 d33)
(smaller peg2 d21)
(smaller peg3 d87)
(smaller peg3 d33)
(smaller peg3 d21)
(smaller d33 d87)
(smaller d21 d87)
(smaller d21 d33)
(clear peg2)
(clear peg3)
(clear d87)
(on d21 peg1)
(on d33 d21)
(on d87 d33)
)
(:goal
(and 
(on d21 peg3)
(on d33 d21)
(on d87 d33)
)
)
)


