


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d296 d408 )
(:init
(smaller peg1 d296)
(smaller peg1 d408)
(smaller peg2 d296)
(smaller peg2 d408)
(smaller peg3 d296)
(smaller peg3 d408)
(smaller d408 d296)
(clear peg2)
(clear peg3)
(clear d296)
(on d408 peg1)
(on d296 d408)
)
(:goal
(and 
(on d408 peg3)
(on d296 d408)
)
)
)


