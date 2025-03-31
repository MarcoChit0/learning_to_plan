


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d332 d146 d257 d60 )
(:init
(smaller peg1 d332)
(smaller peg1 d146)
(smaller peg1 d257)
(smaller peg1 d60)
(smaller peg2 d332)
(smaller peg2 d146)
(smaller peg2 d257)
(smaller peg2 d60)
(smaller peg3 d332)
(smaller peg3 d146)
(smaller peg3 d257)
(smaller peg3 d60)
(smaller d146 d332)
(smaller d257 d332)
(smaller d60 d332)
(smaller d257 d146)
(smaller d60 d146)
(smaller d60 d257)
(clear peg2)
(clear peg3)
(clear d332)
(on d60 peg1)
(on d257 d60)
(on d146 d257)
(on d332 d146)
)
(:goal
(and 
(on d60 peg3)
(on d257 d60)
(on d146 d257)
(on d332 d146)
)
)
)


