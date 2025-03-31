


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d383 )
(:init
(smaller peg1 d383)
(smaller peg2 d383)
(smaller peg3 d383)
(clear peg2)
(clear peg3)
(clear d383)
(on d383 peg1)
)
(:goal
(and 
(on d383 peg3)
)
)
)


