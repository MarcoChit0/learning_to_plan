


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d153 d475 d252 d86 )
(:init
(smaller peg1 d153)
(smaller peg1 d475)
(smaller peg1 d252)
(smaller peg1 d86)
(smaller peg2 d153)
(smaller peg2 d475)
(smaller peg2 d252)
(smaller peg2 d86)
(smaller peg3 d153)
(smaller peg3 d475)
(smaller peg3 d252)
(smaller peg3 d86)
(smaller d475 d153)
(smaller d252 d153)
(smaller d86 d153)
(smaller d252 d475)
(smaller d86 d475)
(smaller d86 d252)
(clear peg2)
(clear peg3)
(clear d153)
(on d86 peg1)
(on d252 d86)
(on d475 d252)
(on d153 d475)
)
(:goal
(and 
(on d86 peg3)
(on d252 d86)
(on d475 d252)
(on d153 d475)
)
)
)


