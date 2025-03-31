


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d74 d184 d397 d302 )
(:init
(smaller peg1 d74)
(smaller peg1 d184)
(smaller peg1 d397)
(smaller peg1 d302)
(smaller peg2 d74)
(smaller peg2 d184)
(smaller peg2 d397)
(smaller peg2 d302)
(smaller peg3 d74)
(smaller peg3 d184)
(smaller peg3 d397)
(smaller peg3 d302)
(smaller d184 d74)
(smaller d397 d74)
(smaller d302 d74)
(smaller d397 d184)
(smaller d302 d184)
(smaller d302 d397)
(clear peg2)
(clear peg3)
(clear d74)
(on d302 peg1)
(on d397 d302)
(on d184 d397)
(on d74 d184)
)
(:goal
(and 
(on d302 peg3)
(on d397 d302)
(on d184 d397)
(on d74 d184)
)
)
)


