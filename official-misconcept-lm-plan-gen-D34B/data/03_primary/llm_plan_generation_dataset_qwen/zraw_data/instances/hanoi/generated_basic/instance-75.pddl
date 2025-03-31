


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d321 d396 )
(:init
(smaller peg1 d321)
(smaller peg1 d396)
(smaller peg2 d321)
(smaller peg2 d396)
(smaller peg3 d321)
(smaller peg3 d396)
(smaller d396 d321)
(clear peg2)
(clear peg3)
(clear d321)
(on d396 peg1)
(on d321 d396)
)
(:goal
(and 
(on d396 peg3)
(on d321 d396)
)
)
)


