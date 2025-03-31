


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d182 d440 )
(:init
(smaller peg1 d182)
(smaller peg1 d440)
(smaller peg2 d182)
(smaller peg2 d440)
(smaller peg3 d182)
(smaller peg3 d440)
(smaller d440 d182)
(clear peg2)
(clear peg3)
(clear d182)
(on d440 peg1)
(on d182 d440)
)
(:goal
(and 
(on d440 peg3)
(on d182 d440)
)
)
)


