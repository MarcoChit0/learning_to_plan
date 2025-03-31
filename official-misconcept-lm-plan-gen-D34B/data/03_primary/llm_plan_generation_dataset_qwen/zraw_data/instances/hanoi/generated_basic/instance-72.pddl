


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d263 d148 )
(:init
(smaller peg1 d263)
(smaller peg1 d148)
(smaller peg2 d263)
(smaller peg2 d148)
(smaller peg3 d263)
(smaller peg3 d148)
(smaller d148 d263)
(clear peg2)
(clear peg3)
(clear d263)
(on d148 peg1)
(on d263 d148)
)
(:goal
(and 
(on d148 peg3)
(on d263 d148)
)
)
)


