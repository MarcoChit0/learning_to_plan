


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d334 d382 )
(:init
(smaller peg1 d334)
(smaller peg1 d382)
(smaller peg2 d334)
(smaller peg2 d382)
(smaller peg3 d334)
(smaller peg3 d382)
(smaller d382 d334)
(clear peg2)
(clear peg3)
(clear d334)
(on d382 peg1)
(on d334 d382)
)
(:goal
(and 
(on d382 peg3)
(on d334 d382)
)
)
)


