


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d458 d163 )
(:init
(smaller peg1 d458)
(smaller peg1 d163)
(smaller peg2 d458)
(smaller peg2 d163)
(smaller peg3 d458)
(smaller peg3 d163)
(smaller d163 d458)
(clear peg2)
(clear peg3)
(clear d458)
(on d163 peg1)
(on d458 d163)
)
(:goal
(and 
(on d163 peg3)
(on d458 d163)
)
)
)


