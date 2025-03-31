


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d424 d481 d464 )
(:init
(smaller peg1 d424)
(smaller peg1 d481)
(smaller peg1 d464)
(smaller peg2 d424)
(smaller peg2 d481)
(smaller peg2 d464)
(smaller peg3 d424)
(smaller peg3 d481)
(smaller peg3 d464)
(smaller d481 d424)
(smaller d464 d424)
(smaller d464 d481)
(clear peg2)
(clear peg3)
(clear d424)
(on d464 peg1)
(on d481 d464)
(on d424 d481)
)
(:goal
(and 
(on d464 peg3)
(on d481 d464)
(on d424 d481)
)
)
)


