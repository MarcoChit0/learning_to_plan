


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d266 d76 )
(:init
(smaller peg1 d266)
(smaller peg1 d76)
(smaller peg2 d266)
(smaller peg2 d76)
(smaller peg3 d266)
(smaller peg3 d76)
(smaller d76 d266)
(clear peg2)
(clear peg3)
(clear d266)
(on d76 peg1)
(on d266 d76)
)
(:goal
(and 
(on d76 peg3)
(on d266 d76)
)
)
)


