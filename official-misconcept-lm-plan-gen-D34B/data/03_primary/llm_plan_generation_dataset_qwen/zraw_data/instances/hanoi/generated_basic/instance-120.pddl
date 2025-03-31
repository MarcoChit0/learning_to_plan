


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d165 d467 d57 )
(:init
(smaller peg1 d165)
(smaller peg1 d467)
(smaller peg1 d57)
(smaller peg2 d165)
(smaller peg2 d467)
(smaller peg2 d57)
(smaller peg3 d165)
(smaller peg3 d467)
(smaller peg3 d57)
(smaller d467 d165)
(smaller d57 d165)
(smaller d57 d467)
(clear peg2)
(clear peg3)
(clear d165)
(on d57 peg1)
(on d467 d57)
(on d165 d467)
)
(:goal
(and 
(on d57 peg3)
(on d467 d57)
(on d165 d467)
)
)
)


