


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d217 d216 )
(:init
(smaller peg1 d217)
(smaller peg1 d216)
(smaller peg2 d217)
(smaller peg2 d216)
(smaller peg3 d217)
(smaller peg3 d216)
(smaller d216 d217)
(clear peg2)
(clear peg3)
(clear d217)
(on d216 peg1)
(on d217 d216)
)
(:goal
(and 
(on d216 peg3)
(on d217 d216)
)
)
)


