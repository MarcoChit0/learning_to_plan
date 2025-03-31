


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d380 d46 )
(:init
(smaller peg1 d380)
(smaller peg1 d46)
(smaller peg2 d380)
(smaller peg2 d46)
(smaller peg3 d380)
(smaller peg3 d46)
(smaller d46 d380)
(clear peg2)
(clear peg3)
(clear d380)
(on d46 peg1)
(on d380 d46)
)
(:goal
(and 
(on d46 peg3)
(on d380 d46)
)
)
)


