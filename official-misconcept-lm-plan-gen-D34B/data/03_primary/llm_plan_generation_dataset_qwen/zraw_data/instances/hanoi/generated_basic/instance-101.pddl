


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d243 d481 d302 )
(:init
(smaller peg1 d243)
(smaller peg1 d481)
(smaller peg1 d302)
(smaller peg2 d243)
(smaller peg2 d481)
(smaller peg2 d302)
(smaller peg3 d243)
(smaller peg3 d481)
(smaller peg3 d302)
(smaller d481 d243)
(smaller d302 d243)
(smaller d302 d481)
(clear peg2)
(clear peg3)
(clear d243)
(on d302 peg1)
(on d481 d302)
(on d243 d481)
)
(:goal
(and 
(on d302 peg3)
(on d481 d302)
(on d243 d481)
)
)
)


