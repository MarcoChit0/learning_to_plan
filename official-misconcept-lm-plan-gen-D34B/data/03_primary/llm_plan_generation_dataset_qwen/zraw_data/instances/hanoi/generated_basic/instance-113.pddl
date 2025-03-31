


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d42 d415 d313 )
(:init
(smaller peg1 d42)
(smaller peg1 d415)
(smaller peg1 d313)
(smaller peg2 d42)
(smaller peg2 d415)
(smaller peg2 d313)
(smaller peg3 d42)
(smaller peg3 d415)
(smaller peg3 d313)
(smaller d415 d42)
(smaller d313 d42)
(smaller d313 d415)
(clear peg2)
(clear peg3)
(clear d42)
(on d313 peg1)
(on d415 d313)
(on d42 d415)
)
(:goal
(and 
(on d313 peg3)
(on d415 d313)
(on d42 d415)
)
)
)


