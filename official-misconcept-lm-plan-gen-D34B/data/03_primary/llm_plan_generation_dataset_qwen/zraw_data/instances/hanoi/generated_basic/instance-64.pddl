


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d214 d283 )
(:init
(smaller peg1 d214)
(smaller peg1 d283)
(smaller peg2 d214)
(smaller peg2 d283)
(smaller peg3 d214)
(smaller peg3 d283)
(smaller d283 d214)
(clear peg2)
(clear peg3)
(clear d214)
(on d283 peg1)
(on d214 d283)
)
(:goal
(and 
(on d283 peg3)
(on d214 d283)
)
)
)


