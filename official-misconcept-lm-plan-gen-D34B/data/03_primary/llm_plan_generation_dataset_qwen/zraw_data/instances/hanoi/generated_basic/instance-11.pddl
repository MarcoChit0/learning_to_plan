


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d425 )
(:init
(smaller peg1 d425)
(smaller peg2 d425)
(smaller peg3 d425)
(clear peg2)
(clear peg3)
(clear d425)
(on d425 peg1)
)
(:goal
(and 
(on d425 peg3)
)
)
)


