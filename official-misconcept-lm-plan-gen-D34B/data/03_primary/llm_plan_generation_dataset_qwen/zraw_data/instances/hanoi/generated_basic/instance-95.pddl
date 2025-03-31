


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d317 d16 )
(:init
(smaller peg1 d317)
(smaller peg1 d16)
(smaller peg2 d317)
(smaller peg2 d16)
(smaller peg3 d317)
(smaller peg3 d16)
(smaller d16 d317)
(clear peg2)
(clear peg3)
(clear d317)
(on d16 peg1)
(on d317 d16)
)
(:goal
(and 
(on d16 peg3)
(on d317 d16)
)
)
)


