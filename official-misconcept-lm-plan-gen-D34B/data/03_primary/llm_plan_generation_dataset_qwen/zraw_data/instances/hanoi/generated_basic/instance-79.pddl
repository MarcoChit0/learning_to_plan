


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d104 d423 )
(:init
(smaller peg1 d104)
(smaller peg1 d423)
(smaller peg2 d104)
(smaller peg2 d423)
(smaller peg3 d104)
(smaller peg3 d423)
(smaller d423 d104)
(clear peg2)
(clear peg3)
(clear d104)
(on d423 peg1)
(on d104 d423)
)
(:goal
(and 
(on d423 peg3)
(on d104 d423)
)
)
)


