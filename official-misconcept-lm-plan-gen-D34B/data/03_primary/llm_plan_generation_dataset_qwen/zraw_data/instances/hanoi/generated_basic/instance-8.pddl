


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d473 )
(:init
(smaller peg1 d473)
(smaller peg2 d473)
(smaller peg3 d473)
(clear peg2)
(clear peg3)
(clear d473)
(on d473 peg1)
)
(:goal
(and 
(on d473 peg3)
)
)
)


