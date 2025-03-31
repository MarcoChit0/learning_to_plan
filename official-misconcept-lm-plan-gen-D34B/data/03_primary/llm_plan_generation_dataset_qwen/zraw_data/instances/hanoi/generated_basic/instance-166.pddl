


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d451 d89 d81 d208 )
(:init
(smaller peg1 d451)
(smaller peg1 d89)
(smaller peg1 d81)
(smaller peg1 d208)
(smaller peg2 d451)
(smaller peg2 d89)
(smaller peg2 d81)
(smaller peg2 d208)
(smaller peg3 d451)
(smaller peg3 d89)
(smaller peg3 d81)
(smaller peg3 d208)
(smaller d89 d451)
(smaller d81 d451)
(smaller d208 d451)
(smaller d81 d89)
(smaller d208 d89)
(smaller d208 d81)
(clear peg2)
(clear peg3)
(clear d451)
(on d208 peg1)
(on d81 d208)
(on d89 d81)
(on d451 d89)
)
(:goal
(and 
(on d208 peg3)
(on d81 d208)
(on d89 d81)
(on d451 d89)
)
)
)


