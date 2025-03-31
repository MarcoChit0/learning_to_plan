


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d356 d478 d294 d397 )
(:init
(smaller peg1 d356)
(smaller peg1 d478)
(smaller peg1 d294)
(smaller peg1 d397)
(smaller peg2 d356)
(smaller peg2 d478)
(smaller peg2 d294)
(smaller peg2 d397)
(smaller peg3 d356)
(smaller peg3 d478)
(smaller peg3 d294)
(smaller peg3 d397)
(smaller d478 d356)
(smaller d294 d356)
(smaller d397 d356)
(smaller d294 d478)
(smaller d397 d478)
(smaller d397 d294)
(clear peg2)
(clear peg3)
(clear d356)
(on d397 peg1)
(on d294 d397)
(on d478 d294)
(on d356 d478)
)
(:goal
(and 
(on d397 peg3)
(on d294 d397)
(on d478 d294)
(on d356 d478)
)
)
)


