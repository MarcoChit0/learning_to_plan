


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d482 d391 )
(:init
(smaller peg1 d482)
(smaller peg1 d391)
(smaller peg2 d482)
(smaller peg2 d391)
(smaller peg3 d482)
(smaller peg3 d391)
(smaller d391 d482)
(clear peg2)
(clear peg3)
(clear d482)
(on d391 peg1)
(on d482 d391)
)
(:goal
(and 
(on d391 peg3)
(on d482 d391)
)
)
)


