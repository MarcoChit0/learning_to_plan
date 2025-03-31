


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d340 d232 d254 )
(:init
(smaller peg1 d340)
(smaller peg1 d232)
(smaller peg1 d254)
(smaller peg2 d340)
(smaller peg2 d232)
(smaller peg2 d254)
(smaller peg3 d340)
(smaller peg3 d232)
(smaller peg3 d254)
(smaller d232 d340)
(smaller d254 d340)
(smaller d254 d232)
(clear peg2)
(clear peg3)
(clear d340)
(on d254 peg1)
(on d232 d254)
(on d340 d232)
)
(:goal
(and 
(on d254 peg3)
(on d232 d254)
(on d340 d232)
)
)
)


