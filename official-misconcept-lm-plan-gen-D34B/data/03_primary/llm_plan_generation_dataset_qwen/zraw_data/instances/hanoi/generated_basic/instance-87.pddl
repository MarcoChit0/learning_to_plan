


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d397 d288 )
(:init
(smaller peg1 d397)
(smaller peg1 d288)
(smaller peg2 d397)
(smaller peg2 d288)
(smaller peg3 d397)
(smaller peg3 d288)
(smaller d288 d397)
(clear peg2)
(clear peg3)
(clear d397)
(on d288 peg1)
(on d397 d288)
)
(:goal
(and 
(on d288 peg3)
(on d397 d288)
)
)
)


