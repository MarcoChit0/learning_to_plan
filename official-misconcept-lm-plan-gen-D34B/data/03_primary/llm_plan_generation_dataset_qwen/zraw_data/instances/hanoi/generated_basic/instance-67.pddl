


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d86 d240 )
(:init
(smaller peg1 d86)
(smaller peg1 d240)
(smaller peg2 d86)
(smaller peg2 d240)
(smaller peg3 d86)
(smaller peg3 d240)
(smaller d240 d86)
(clear peg2)
(clear peg3)
(clear d86)
(on d240 peg1)
(on d86 d240)
)
(:goal
(and 
(on d240 peg3)
(on d86 d240)
)
)
)


