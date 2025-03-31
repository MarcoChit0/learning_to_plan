


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d332 d304 )
(:init
(smaller peg1 d332)
(smaller peg1 d304)
(smaller peg2 d332)
(smaller peg2 d304)
(smaller peg3 d332)
(smaller peg3 d304)
(smaller d304 d332)
(clear peg2)
(clear peg3)
(clear d332)
(on d304 peg1)
(on d332 d304)
)
(:goal
(and 
(on d304 peg3)
(on d332 d304)
)
)
)


