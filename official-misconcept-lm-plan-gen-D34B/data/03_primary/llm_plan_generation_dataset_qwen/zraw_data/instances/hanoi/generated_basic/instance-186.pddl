


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d325 d408 d348 d107 )
(:init
(smaller peg1 d325)
(smaller peg1 d408)
(smaller peg1 d348)
(smaller peg1 d107)
(smaller peg2 d325)
(smaller peg2 d408)
(smaller peg2 d348)
(smaller peg2 d107)
(smaller peg3 d325)
(smaller peg3 d408)
(smaller peg3 d348)
(smaller peg3 d107)
(smaller d408 d325)
(smaller d348 d325)
(smaller d107 d325)
(smaller d348 d408)
(smaller d107 d408)
(smaller d107 d348)
(clear peg2)
(clear peg3)
(clear d325)
(on d107 peg1)
(on d348 d107)
(on d408 d348)
(on d325 d408)
)
(:goal
(and 
(on d107 peg3)
(on d348 d107)
(on d408 d348)
(on d325 d408)
)
)
)


