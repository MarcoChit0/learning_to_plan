


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d253 d13 d183 )
(:init
(smaller peg1 d253)
(smaller peg1 d13)
(smaller peg1 d183)
(smaller peg2 d253)
(smaller peg2 d13)
(smaller peg2 d183)
(smaller peg3 d253)
(smaller peg3 d13)
(smaller peg3 d183)
(smaller d13 d253)
(smaller d183 d253)
(smaller d183 d13)
(clear peg2)
(clear peg3)
(clear d253)
(on d183 peg1)
(on d13 d183)
(on d253 d13)
)
(:goal
(and 
(on d183 peg3)
(on d13 d183)
(on d253 d13)
)
)
)


