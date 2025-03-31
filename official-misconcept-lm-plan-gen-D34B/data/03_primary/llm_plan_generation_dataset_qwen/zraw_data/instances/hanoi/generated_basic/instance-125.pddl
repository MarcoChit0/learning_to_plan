


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d410 d397 d42 )
(:init
(smaller peg1 d410)
(smaller peg1 d397)
(smaller peg1 d42)
(smaller peg2 d410)
(smaller peg2 d397)
(smaller peg2 d42)
(smaller peg3 d410)
(smaller peg3 d397)
(smaller peg3 d42)
(smaller d397 d410)
(smaller d42 d410)
(smaller d42 d397)
(clear peg2)
(clear peg3)
(clear d410)
(on d42 peg1)
(on d397 d42)
(on d410 d397)
)
(:goal
(and 
(on d42 peg3)
(on d397 d42)
(on d410 d397)
)
)
)


