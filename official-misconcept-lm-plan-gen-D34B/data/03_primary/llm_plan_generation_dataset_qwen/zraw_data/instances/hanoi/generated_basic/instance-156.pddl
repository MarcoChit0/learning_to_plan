


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d391 d145 d396 d191 )
(:init
(smaller peg1 d391)
(smaller peg1 d145)
(smaller peg1 d396)
(smaller peg1 d191)
(smaller peg2 d391)
(smaller peg2 d145)
(smaller peg2 d396)
(smaller peg2 d191)
(smaller peg3 d391)
(smaller peg3 d145)
(smaller peg3 d396)
(smaller peg3 d191)
(smaller d145 d391)
(smaller d396 d391)
(smaller d191 d391)
(smaller d396 d145)
(smaller d191 d145)
(smaller d191 d396)
(clear peg2)
(clear peg3)
(clear d391)
(on d191 peg1)
(on d396 d191)
(on d145 d396)
(on d391 d145)
)
(:goal
(and 
(on d191 peg3)
(on d396 d191)
(on d145 d396)
(on d391 d145)
)
)
)


