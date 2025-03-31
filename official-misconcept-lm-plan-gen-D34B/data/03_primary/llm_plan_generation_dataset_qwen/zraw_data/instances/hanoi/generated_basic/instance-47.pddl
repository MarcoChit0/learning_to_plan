


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d274 )
(:init
(smaller peg1 d274)
(smaller peg2 d274)
(smaller peg3 d274)
(clear peg2)
(clear peg3)
(clear d274)
(on d274 peg1)
)
(:goal
(and 
(on d274 peg3)
)
)
)


