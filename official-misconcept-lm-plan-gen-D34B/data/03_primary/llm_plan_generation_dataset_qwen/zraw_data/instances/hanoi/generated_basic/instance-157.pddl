


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d146 d485 d374 d98 )
(:init
(smaller peg1 d146)
(smaller peg1 d485)
(smaller peg1 d374)
(smaller peg1 d98)
(smaller peg2 d146)
(smaller peg2 d485)
(smaller peg2 d374)
(smaller peg2 d98)
(smaller peg3 d146)
(smaller peg3 d485)
(smaller peg3 d374)
(smaller peg3 d98)
(smaller d485 d146)
(smaller d374 d146)
(smaller d98 d146)
(smaller d374 d485)
(smaller d98 d485)
(smaller d98 d374)
(clear peg2)
(clear peg3)
(clear d146)
(on d98 peg1)
(on d374 d98)
(on d485 d374)
(on d146 d485)
)
(:goal
(and 
(on d98 peg3)
(on d374 d98)
(on d485 d374)
(on d146 d485)
)
)
)


