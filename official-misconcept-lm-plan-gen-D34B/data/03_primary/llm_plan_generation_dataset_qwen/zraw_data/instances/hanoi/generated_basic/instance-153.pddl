


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d38 d323 d438 d72 )
(:init
(smaller peg1 d38)
(smaller peg1 d323)
(smaller peg1 d438)
(smaller peg1 d72)
(smaller peg2 d38)
(smaller peg2 d323)
(smaller peg2 d438)
(smaller peg2 d72)
(smaller peg3 d38)
(smaller peg3 d323)
(smaller peg3 d438)
(smaller peg3 d72)
(smaller d323 d38)
(smaller d438 d38)
(smaller d72 d38)
(smaller d438 d323)
(smaller d72 d323)
(smaller d72 d438)
(clear peg2)
(clear peg3)
(clear d38)
(on d72 peg1)
(on d438 d72)
(on d323 d438)
(on d38 d323)
)
(:goal
(and 
(on d72 peg3)
(on d438 d72)
(on d323 d438)
(on d38 d323)
)
)
)


