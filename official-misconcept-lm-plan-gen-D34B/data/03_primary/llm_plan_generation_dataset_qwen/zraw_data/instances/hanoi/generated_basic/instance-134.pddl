


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d33 d422 d128 )
(:init
(smaller peg1 d33)
(smaller peg1 d422)
(smaller peg1 d128)
(smaller peg2 d33)
(smaller peg2 d422)
(smaller peg2 d128)
(smaller peg3 d33)
(smaller peg3 d422)
(smaller peg3 d128)
(smaller d422 d33)
(smaller d128 d33)
(smaller d128 d422)
(clear peg2)
(clear peg3)
(clear d33)
(on d128 peg1)
(on d422 d128)
(on d33 d422)
)
(:goal
(and 
(on d128 peg3)
(on d422 d128)
(on d33 d422)
)
)
)


