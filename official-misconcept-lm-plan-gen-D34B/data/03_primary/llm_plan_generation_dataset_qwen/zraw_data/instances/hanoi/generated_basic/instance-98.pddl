


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d475 d146 )
(:init
(smaller peg1 d475)
(smaller peg1 d146)
(smaller peg2 d475)
(smaller peg2 d146)
(smaller peg3 d475)
(smaller peg3 d146)
(smaller d146 d475)
(clear peg2)
(clear peg3)
(clear d475)
(on d146 peg1)
(on d475 d146)
)
(:goal
(and 
(on d146 peg3)
(on d475 d146)
)
)
)


