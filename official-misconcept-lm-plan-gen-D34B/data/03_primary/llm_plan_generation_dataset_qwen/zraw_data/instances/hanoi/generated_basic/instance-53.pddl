


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d89 d333 )
(:init
(smaller peg1 d89)
(smaller peg1 d333)
(smaller peg2 d89)
(smaller peg2 d333)
(smaller peg3 d89)
(smaller peg3 d333)
(smaller d333 d89)
(clear peg2)
(clear peg3)
(clear d89)
(on d333 peg1)
(on d89 d333)
)
(:goal
(and 
(on d333 peg3)
(on d89 d333)
)
)
)


