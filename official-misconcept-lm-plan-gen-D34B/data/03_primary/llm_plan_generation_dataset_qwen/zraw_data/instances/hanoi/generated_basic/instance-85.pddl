


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d70 d348 )
(:init
(smaller peg1 d70)
(smaller peg1 d348)
(smaller peg2 d70)
(smaller peg2 d348)
(smaller peg3 d70)
(smaller peg3 d348)
(smaller d348 d70)
(clear peg2)
(clear peg3)
(clear d70)
(on d348 peg1)
(on d70 d348)
)
(:goal
(and 
(on d348 peg3)
(on d70 d348)
)
)
)


