


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d140 d91 d56 )
(:init
(smaller peg1 d140)
(smaller peg1 d91)
(smaller peg1 d56)
(smaller peg2 d140)
(smaller peg2 d91)
(smaller peg2 d56)
(smaller peg3 d140)
(smaller peg3 d91)
(smaller peg3 d56)
(smaller d91 d140)
(smaller d56 d140)
(smaller d56 d91)
(clear peg2)
(clear peg3)
(clear d140)
(on d56 peg1)
(on d91 d56)
(on d140 d91)
)
(:goal
(and 
(on d56 peg3)
(on d91 d56)
(on d140 d91)
)
)
)


