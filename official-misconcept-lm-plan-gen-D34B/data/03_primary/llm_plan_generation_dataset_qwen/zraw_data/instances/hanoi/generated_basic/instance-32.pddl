


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d268 )
(:init
(smaller peg1 d268)
(smaller peg2 d268)
(smaller peg3 d268)
(clear peg2)
(clear peg3)
(clear d268)
(on d268 peg1)
)
(:goal
(and 
(on d268 peg3)
)
)
)


