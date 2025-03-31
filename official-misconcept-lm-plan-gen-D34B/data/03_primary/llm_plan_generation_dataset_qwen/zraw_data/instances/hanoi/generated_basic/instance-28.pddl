


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d423 )
(:init
(smaller peg1 d423)
(smaller peg2 d423)
(smaller peg3 d423)
(clear peg2)
(clear peg3)
(clear d423)
(on d423 peg1)
)
(:goal
(and 
(on d423 peg3)
)
)
)


