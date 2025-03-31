


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d442 d243 )
(:init
(smaller peg1 d442)
(smaller peg1 d243)
(smaller peg2 d442)
(smaller peg2 d243)
(smaller peg3 d442)
(smaller peg3 d243)
(smaller d243 d442)
(clear peg2)
(clear peg3)
(clear d442)
(on d243 peg1)
(on d442 d243)
)
(:goal
(and 
(on d243 peg3)
(on d442 d243)
)
)
)


