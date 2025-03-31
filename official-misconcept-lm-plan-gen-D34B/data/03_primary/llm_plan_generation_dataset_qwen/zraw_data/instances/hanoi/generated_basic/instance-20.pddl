


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d38 )
(:init
(smaller peg1 d38)
(smaller peg2 d38)
(smaller peg3 d38)
(clear peg2)
(clear peg3)
(clear d38)
(on d38 peg1)
)
(:goal
(and 
(on d38 peg3)
)
)
)


