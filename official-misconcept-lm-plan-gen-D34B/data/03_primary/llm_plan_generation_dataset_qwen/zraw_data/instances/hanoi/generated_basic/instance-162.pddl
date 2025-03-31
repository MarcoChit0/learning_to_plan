


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d10 d375 d141 d368 )
(:init
(smaller peg1 d10)
(smaller peg1 d375)
(smaller peg1 d141)
(smaller peg1 d368)
(smaller peg2 d10)
(smaller peg2 d375)
(smaller peg2 d141)
(smaller peg2 d368)
(smaller peg3 d10)
(smaller peg3 d375)
(smaller peg3 d141)
(smaller peg3 d368)
(smaller d375 d10)
(smaller d141 d10)
(smaller d368 d10)
(smaller d141 d375)
(smaller d368 d375)
(smaller d368 d141)
(clear peg2)
(clear peg3)
(clear d10)
(on d368 peg1)
(on d141 d368)
(on d375 d141)
(on d10 d375)
)
(:goal
(and 
(on d368 peg3)
(on d141 d368)
(on d375 d141)
(on d10 d375)
)
)
)


