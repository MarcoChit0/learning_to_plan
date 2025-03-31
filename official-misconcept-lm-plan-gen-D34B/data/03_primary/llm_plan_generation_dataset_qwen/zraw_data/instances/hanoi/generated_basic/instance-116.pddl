


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d486 d19 d307 )
(:init
(smaller peg1 d486)
(smaller peg1 d19)
(smaller peg1 d307)
(smaller peg2 d486)
(smaller peg2 d19)
(smaller peg2 d307)
(smaller peg3 d486)
(smaller peg3 d19)
(smaller peg3 d307)
(smaller d19 d486)
(smaller d307 d486)
(smaller d307 d19)
(clear peg2)
(clear peg3)
(clear d486)
(on d307 peg1)
(on d19 d307)
(on d486 d19)
)
(:goal
(and 
(on d307 peg3)
(on d19 d307)
(on d486 d19)
)
)
)


