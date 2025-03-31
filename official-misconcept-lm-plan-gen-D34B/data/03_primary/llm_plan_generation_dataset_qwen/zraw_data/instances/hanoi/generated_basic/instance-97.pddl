


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d169 d447 )
(:init
(smaller peg1 d169)
(smaller peg1 d447)
(smaller peg2 d169)
(smaller peg2 d447)
(smaller peg3 d169)
(smaller peg3 d447)
(smaller d447 d169)
(clear peg2)
(clear peg3)
(clear d169)
(on d447 peg1)
(on d169 d447)
)
(:goal
(and 
(on d447 peg3)
(on d169 d447)
)
)
)


