


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d306 d53 d423 )
(:init
(smaller peg1 d306)
(smaller peg1 d53)
(smaller peg1 d423)
(smaller peg2 d306)
(smaller peg2 d53)
(smaller peg2 d423)
(smaller peg3 d306)
(smaller peg3 d53)
(smaller peg3 d423)
(smaller d53 d306)
(smaller d423 d306)
(smaller d423 d53)
(clear peg2)
(clear peg3)
(clear d306)
(on d423 peg1)
(on d53 d423)
(on d306 d53)
)
(:goal
(and 
(on d423 peg3)
(on d53 d423)
(on d306 d53)
)
)
)


