


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d384 d411 )
(:init
(smaller peg1 d384)
(smaller peg1 d411)
(smaller peg2 d384)
(smaller peg2 d411)
(smaller peg3 d384)
(smaller peg3 d411)
(smaller d411 d384)
(clear peg2)
(clear peg3)
(clear d384)
(on d411 peg1)
(on d384 d411)
)
(:goal
(and 
(on d411 peg3)
(on d384 d411)
)
)
)


