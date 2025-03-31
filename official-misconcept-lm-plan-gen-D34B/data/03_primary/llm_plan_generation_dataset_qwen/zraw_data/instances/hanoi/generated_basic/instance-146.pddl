


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d453 d208 d69 )
(:init
(smaller peg1 d453)
(smaller peg1 d208)
(smaller peg1 d69)
(smaller peg2 d453)
(smaller peg2 d208)
(smaller peg2 d69)
(smaller peg3 d453)
(smaller peg3 d208)
(smaller peg3 d69)
(smaller d208 d453)
(smaller d69 d453)
(smaller d69 d208)
(clear peg2)
(clear peg3)
(clear d453)
(on d69 peg1)
(on d208 d69)
(on d453 d208)
)
(:goal
(and 
(on d69 peg3)
(on d208 d69)
(on d453 d208)
)
)
)


