


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d245 d88 d338 )
(:init
(smaller peg1 d245)
(smaller peg1 d88)
(smaller peg1 d338)
(smaller peg2 d245)
(smaller peg2 d88)
(smaller peg2 d338)
(smaller peg3 d245)
(smaller peg3 d88)
(smaller peg3 d338)
(smaller d88 d245)
(smaller d338 d245)
(smaller d338 d88)
(clear peg2)
(clear peg3)
(clear d245)
(on d338 peg1)
(on d88 d338)
(on d245 d88)
)
(:goal
(and 
(on d338 peg3)
(on d88 d338)
(on d245 d88)
)
)
)


