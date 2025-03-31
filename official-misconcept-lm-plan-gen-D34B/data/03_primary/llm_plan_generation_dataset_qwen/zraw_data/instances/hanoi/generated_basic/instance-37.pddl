


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d178 )
(:init
(smaller peg1 d178)
(smaller peg2 d178)
(smaller peg3 d178)
(clear peg2)
(clear peg3)
(clear d178)
(on d178 peg1)
)
(:goal
(and 
(on d178 peg3)
)
)
)


