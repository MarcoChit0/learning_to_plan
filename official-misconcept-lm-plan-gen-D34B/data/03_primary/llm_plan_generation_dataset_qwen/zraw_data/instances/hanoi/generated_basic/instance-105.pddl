


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d258 d316 d302 )
(:init
(smaller peg1 d258)
(smaller peg1 d316)
(smaller peg1 d302)
(smaller peg2 d258)
(smaller peg2 d316)
(smaller peg2 d302)
(smaller peg3 d258)
(smaller peg3 d316)
(smaller peg3 d302)
(smaller d316 d258)
(smaller d302 d258)
(smaller d302 d316)
(clear peg2)
(clear peg3)
(clear d258)
(on d302 peg1)
(on d316 d302)
(on d258 d316)
)
(:goal
(and 
(on d302 peg3)
(on d316 d302)
(on d258 d316)
)
)
)


