


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d160 d44 )
(:init
(smaller peg1 d160)
(smaller peg1 d44)
(smaller peg2 d160)
(smaller peg2 d44)
(smaller peg3 d160)
(smaller peg3 d44)
(smaller d44 d160)
(clear peg2)
(clear peg3)
(clear d160)
(on d44 peg1)
(on d160 d44)
)
(:goal
(and 
(on d44 peg3)
(on d160 d44)
)
)
)


