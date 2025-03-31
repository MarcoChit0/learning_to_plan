


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d249 d379 d481 )
(:init
(smaller peg1 d249)
(smaller peg1 d379)
(smaller peg1 d481)
(smaller peg2 d249)
(smaller peg2 d379)
(smaller peg2 d481)
(smaller peg3 d249)
(smaller peg3 d379)
(smaller peg3 d481)
(smaller d379 d249)
(smaller d481 d249)
(smaller d481 d379)
(clear peg2)
(clear peg3)
(clear d249)
(on d481 peg1)
(on d379 d481)
(on d249 d379)
)
(:goal
(and 
(on d481 peg3)
(on d379 d481)
(on d249 d379)
)
)
)


