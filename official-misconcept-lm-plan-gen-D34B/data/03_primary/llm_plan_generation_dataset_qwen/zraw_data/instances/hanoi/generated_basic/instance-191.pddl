


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d86 d394 d87 d396 )
(:init
(smaller peg1 d86)
(smaller peg1 d394)
(smaller peg1 d87)
(smaller peg1 d396)
(smaller peg2 d86)
(smaller peg2 d394)
(smaller peg2 d87)
(smaller peg2 d396)
(smaller peg3 d86)
(smaller peg3 d394)
(smaller peg3 d87)
(smaller peg3 d396)
(smaller d394 d86)
(smaller d87 d86)
(smaller d396 d86)
(smaller d87 d394)
(smaller d396 d394)
(smaller d396 d87)
(clear peg2)
(clear peg3)
(clear d86)
(on d396 peg1)
(on d87 d396)
(on d394 d87)
(on d86 d394)
)
(:goal
(and 
(on d396 peg3)
(on d87 d396)
(on d394 d87)
(on d86 d394)
)
)
)


