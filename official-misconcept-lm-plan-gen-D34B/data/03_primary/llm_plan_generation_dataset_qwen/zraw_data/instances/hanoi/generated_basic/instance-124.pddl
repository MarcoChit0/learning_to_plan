


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d166 d454 d179 )
(:init
(smaller peg1 d166)
(smaller peg1 d454)
(smaller peg1 d179)
(smaller peg2 d166)
(smaller peg2 d454)
(smaller peg2 d179)
(smaller peg3 d166)
(smaller peg3 d454)
(smaller peg3 d179)
(smaller d454 d166)
(smaller d179 d166)
(smaller d179 d454)
(clear peg2)
(clear peg3)
(clear d166)
(on d179 peg1)
(on d454 d179)
(on d166 d454)
)
(:goal
(and 
(on d179 peg3)
(on d454 d179)
(on d166 d454)
)
)
)


