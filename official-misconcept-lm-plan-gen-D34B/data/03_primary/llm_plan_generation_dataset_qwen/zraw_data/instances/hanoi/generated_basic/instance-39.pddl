


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d420 )
(:init
(smaller peg1 d420)
(smaller peg2 d420)
(smaller peg3 d420)
(clear peg2)
(clear peg3)
(clear d420)
(on d420 peg1)
)
(:goal
(and 
(on d420 peg3)
)
)
)


