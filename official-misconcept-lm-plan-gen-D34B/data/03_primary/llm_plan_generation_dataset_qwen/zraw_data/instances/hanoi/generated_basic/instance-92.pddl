


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d463 d238 )
(:init
(smaller peg1 d463)
(smaller peg1 d238)
(smaller peg2 d463)
(smaller peg2 d238)
(smaller peg3 d463)
(smaller peg3 d238)
(smaller d238 d463)
(clear peg2)
(clear peg3)
(clear d463)
(on d238 peg1)
(on d463 d238)
)
(:goal
(and 
(on d238 peg3)
(on d463 d238)
)
)
)


