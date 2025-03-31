


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d250 d309 d416 d463 )
(:init
(smaller peg1 d250)
(smaller peg1 d309)
(smaller peg1 d416)
(smaller peg1 d463)
(smaller peg2 d250)
(smaller peg2 d309)
(smaller peg2 d416)
(smaller peg2 d463)
(smaller peg3 d250)
(smaller peg3 d309)
(smaller peg3 d416)
(smaller peg3 d463)
(smaller d309 d250)
(smaller d416 d250)
(smaller d463 d250)
(smaller d416 d309)
(smaller d463 d309)
(smaller d463 d416)
(clear peg2)
(clear peg3)
(clear d250)
(on d463 peg1)
(on d416 d463)
(on d309 d416)
(on d250 d309)
)
(:goal
(and 
(on d463 peg3)
(on d416 d463)
(on d309 d416)
(on d250 d309)
)
)
)


