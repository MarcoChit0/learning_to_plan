


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d248 d108 )
(:init
(smaller peg1 d248)
(smaller peg1 d108)
(smaller peg2 d248)
(smaller peg2 d108)
(smaller peg3 d248)
(smaller peg3 d108)
(smaller d108 d248)
(clear peg2)
(clear peg3)
(clear d248)
(on d108 peg1)
(on d248 d108)
)
(:goal
(and 
(on d108 peg3)
(on d248 d108)
)
)
)


