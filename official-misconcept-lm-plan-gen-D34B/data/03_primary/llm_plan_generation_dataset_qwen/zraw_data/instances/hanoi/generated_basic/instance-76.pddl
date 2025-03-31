


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d31 d110 )
(:init
(smaller peg1 d31)
(smaller peg1 d110)
(smaller peg2 d31)
(smaller peg2 d110)
(smaller peg3 d31)
(smaller peg3 d110)
(smaller d110 d31)
(clear peg2)
(clear peg3)
(clear d31)
(on d110 peg1)
(on d31 d110)
)
(:goal
(and 
(on d110 peg3)
(on d31 d110)
)
)
)


