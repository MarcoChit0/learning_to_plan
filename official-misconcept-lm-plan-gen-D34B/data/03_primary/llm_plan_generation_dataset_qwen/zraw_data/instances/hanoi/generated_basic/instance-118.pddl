


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d278 d105 d145 )
(:init
(smaller peg1 d278)
(smaller peg1 d105)
(smaller peg1 d145)
(smaller peg2 d278)
(smaller peg2 d105)
(smaller peg2 d145)
(smaller peg3 d278)
(smaller peg3 d105)
(smaller peg3 d145)
(smaller d105 d278)
(smaller d145 d278)
(smaller d145 d105)
(clear peg2)
(clear peg3)
(clear d278)
(on d145 peg1)
(on d105 d145)
(on d278 d105)
)
(:goal
(and 
(on d145 peg3)
(on d105 d145)
(on d278 d105)
)
)
)


