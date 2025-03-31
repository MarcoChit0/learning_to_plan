


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d225 d243 )
(:init
(smaller peg1 d225)
(smaller peg1 d243)
(smaller peg2 d225)
(smaller peg2 d243)
(smaller peg3 d225)
(smaller peg3 d243)
(smaller d243 d225)
(clear peg2)
(clear peg3)
(clear d225)
(on d243 peg1)
(on d225 d243)
)
(:goal
(and 
(on d243 peg3)
(on d225 d243)
)
)
)


