


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d62 d38 d299 )
(:init
(smaller peg1 d62)
(smaller peg1 d38)
(smaller peg1 d299)
(smaller peg2 d62)
(smaller peg2 d38)
(smaller peg2 d299)
(smaller peg3 d62)
(smaller peg3 d38)
(smaller peg3 d299)
(smaller d38 d62)
(smaller d299 d62)
(smaller d299 d38)
(clear peg2)
(clear peg3)
(clear d62)
(on d299 peg1)
(on d38 d299)
(on d62 d38)
)
(:goal
(and 
(on d299 peg3)
(on d38 d299)
(on d62 d38)
)
)
)


