


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d54 d360 )
(:init
(smaller peg1 d54)
(smaller peg1 d360)
(smaller peg2 d54)
(smaller peg2 d360)
(smaller peg3 d54)
(smaller peg3 d360)
(smaller d360 d54)
(clear peg2)
(clear peg3)
(clear d54)
(on d360 peg1)
(on d54 d360)
)
(:goal
(and 
(on d360 peg3)
(on d54 d360)
)
)
)


