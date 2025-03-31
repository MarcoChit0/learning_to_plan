


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d358 d302 d338 d91 )
(:init
(smaller peg1 d358)
(smaller peg1 d302)
(smaller peg1 d338)
(smaller peg1 d91)
(smaller peg2 d358)
(smaller peg2 d302)
(smaller peg2 d338)
(smaller peg2 d91)
(smaller peg3 d358)
(smaller peg3 d302)
(smaller peg3 d338)
(smaller peg3 d91)
(smaller d302 d358)
(smaller d338 d358)
(smaller d91 d358)
(smaller d338 d302)
(smaller d91 d302)
(smaller d91 d338)
(clear peg2)
(clear peg3)
(clear d358)
(on d91 peg1)
(on d338 d91)
(on d302 d338)
(on d358 d302)
)
(:goal
(and 
(on d91 peg3)
(on d338 d91)
(on d302 d338)
(on d358 d302)
)
)
)


