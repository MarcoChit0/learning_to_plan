


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d448 )
(:init
(smaller peg1 d448)
(smaller peg2 d448)
(smaller peg3 d448)
(clear peg2)
(clear peg3)
(clear d448)
(on d448 peg1)
)
(:goal
(and 
(on d448 peg3)
)
)
)


