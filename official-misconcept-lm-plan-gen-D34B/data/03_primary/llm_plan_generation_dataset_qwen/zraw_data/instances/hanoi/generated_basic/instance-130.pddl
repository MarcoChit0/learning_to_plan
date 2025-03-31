


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d113 d166 d456 )
(:init
(smaller peg1 d113)
(smaller peg1 d166)
(smaller peg1 d456)
(smaller peg2 d113)
(smaller peg2 d166)
(smaller peg2 d456)
(smaller peg3 d113)
(smaller peg3 d166)
(smaller peg3 d456)
(smaller d166 d113)
(smaller d456 d113)
(smaller d456 d166)
(clear peg2)
(clear peg3)
(clear d113)
(on d456 peg1)
(on d166 d456)
(on d113 d166)
)
(:goal
(and 
(on d456 peg3)
(on d166 d456)
(on d113 d166)
)
)
)


