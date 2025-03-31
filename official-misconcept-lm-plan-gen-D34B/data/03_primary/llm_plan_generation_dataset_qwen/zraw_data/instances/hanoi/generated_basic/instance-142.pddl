


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d335 d246 d432 )
(:init
(smaller peg1 d335)
(smaller peg1 d246)
(smaller peg1 d432)
(smaller peg2 d335)
(smaller peg2 d246)
(smaller peg2 d432)
(smaller peg3 d335)
(smaller peg3 d246)
(smaller peg3 d432)
(smaller d246 d335)
(smaller d432 d335)
(smaller d432 d246)
(clear peg2)
(clear peg3)
(clear d335)
(on d432 peg1)
(on d246 d432)
(on d335 d246)
)
(:goal
(and 
(on d432 peg3)
(on d246 d432)
(on d335 d246)
)
)
)


