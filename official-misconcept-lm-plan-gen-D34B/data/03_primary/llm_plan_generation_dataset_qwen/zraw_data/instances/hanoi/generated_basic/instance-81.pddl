


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d417 d294 )
(:init
(smaller peg1 d417)
(smaller peg1 d294)
(smaller peg2 d417)
(smaller peg2 d294)
(smaller peg3 d417)
(smaller peg3 d294)
(smaller d294 d417)
(clear peg2)
(clear peg3)
(clear d417)
(on d294 peg1)
(on d417 d294)
)
(:goal
(and 
(on d294 peg3)
(on d417 d294)
)
)
)


