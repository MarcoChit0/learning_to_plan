


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d135 d238 )
(:init
(smaller peg1 d135)
(smaller peg1 d238)
(smaller peg2 d135)
(smaller peg2 d238)
(smaller peg3 d135)
(smaller peg3 d238)
(smaller d238 d135)
(clear peg2)
(clear peg3)
(clear d135)
(on d238 peg1)
(on d135 d238)
)
(:goal
(and 
(on d238 peg3)
(on d135 d238)
)
)
)


