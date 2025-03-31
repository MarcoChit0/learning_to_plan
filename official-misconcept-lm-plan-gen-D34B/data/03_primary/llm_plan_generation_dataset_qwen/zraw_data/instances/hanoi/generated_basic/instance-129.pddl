


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d91 d7 d419 )
(:init
(smaller peg1 d91)
(smaller peg1 d7)
(smaller peg1 d419)
(smaller peg2 d91)
(smaller peg2 d7)
(smaller peg2 d419)
(smaller peg3 d91)
(smaller peg3 d7)
(smaller peg3 d419)
(smaller d7 d91)
(smaller d419 d91)
(smaller d419 d7)
(clear peg2)
(clear peg3)
(clear d91)
(on d419 peg1)
(on d7 d419)
(on d91 d7)
)
(:goal
(and 
(on d419 peg3)
(on d7 d419)
(on d91 d7)
)
)
)


