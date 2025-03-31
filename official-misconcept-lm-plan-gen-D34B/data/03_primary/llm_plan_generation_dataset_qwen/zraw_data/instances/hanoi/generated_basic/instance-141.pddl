


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d303 d83 d397 )
(:init
(smaller peg1 d303)
(smaller peg1 d83)
(smaller peg1 d397)
(smaller peg2 d303)
(smaller peg2 d83)
(smaller peg2 d397)
(smaller peg3 d303)
(smaller peg3 d83)
(smaller peg3 d397)
(smaller d83 d303)
(smaller d397 d303)
(smaller d397 d83)
(clear peg2)
(clear peg3)
(clear d303)
(on d397 peg1)
(on d83 d397)
(on d303 d83)
)
(:goal
(and 
(on d397 peg3)
(on d83 d397)
(on d303 d83)
)
)
)


