


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d318 d338 )
(:init
(smaller peg1 d318)
(smaller peg1 d338)
(smaller peg2 d318)
(smaller peg2 d338)
(smaller peg3 d318)
(smaller peg3 d338)
(smaller d338 d318)
(clear peg2)
(clear peg3)
(clear d318)
(on d338 peg1)
(on d318 d338)
)
(:goal
(and 
(on d338 peg3)
(on d318 d338)
)
)
)


