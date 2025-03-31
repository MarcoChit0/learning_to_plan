


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d306 d175 )
(:init
(smaller peg1 d306)
(smaller peg1 d175)
(smaller peg2 d306)
(smaller peg2 d175)
(smaller peg3 d306)
(smaller peg3 d175)
(smaller d175 d306)
(clear peg2)
(clear peg3)
(clear d306)
(on d175 peg1)
(on d306 d175)
)
(:goal
(and 
(on d175 peg3)
(on d306 d175)
)
)
)


