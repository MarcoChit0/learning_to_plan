


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d498 d74 d419 d199 )
(:init
(smaller peg1 d498)
(smaller peg1 d74)
(smaller peg1 d419)
(smaller peg1 d199)
(smaller peg2 d498)
(smaller peg2 d74)
(smaller peg2 d419)
(smaller peg2 d199)
(smaller peg3 d498)
(smaller peg3 d74)
(smaller peg3 d419)
(smaller peg3 d199)
(smaller d74 d498)
(smaller d419 d498)
(smaller d199 d498)
(smaller d419 d74)
(smaller d199 d74)
(smaller d199 d419)
(clear peg2)
(clear peg3)
(clear d498)
(on d199 peg1)
(on d419 d199)
(on d74 d419)
(on d498 d74)
)
(:goal
(and 
(on d199 peg3)
(on d419 d199)
(on d74 d419)
(on d498 d74)
)
)
)


