


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d417 d72 d349 d335 )
(:init
(smaller peg1 d417)
(smaller peg1 d72)
(smaller peg1 d349)
(smaller peg1 d335)
(smaller peg2 d417)
(smaller peg2 d72)
(smaller peg2 d349)
(smaller peg2 d335)
(smaller peg3 d417)
(smaller peg3 d72)
(smaller peg3 d349)
(smaller peg3 d335)
(smaller d72 d417)
(smaller d349 d417)
(smaller d335 d417)
(smaller d349 d72)
(smaller d335 d72)
(smaller d335 d349)
(clear peg2)
(clear peg3)
(clear d417)
(on d335 peg1)
(on d349 d335)
(on d72 d349)
(on d417 d72)
)
(:goal
(and 
(on d335 peg3)
(on d349 d335)
(on d72 d349)
(on d417 d72)
)
)
)


