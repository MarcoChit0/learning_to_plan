


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d459 d107 d321 )
(:init
(smaller peg1 d459)
(smaller peg1 d107)
(smaller peg1 d321)
(smaller peg2 d459)
(smaller peg2 d107)
(smaller peg2 d321)
(smaller peg3 d459)
(smaller peg3 d107)
(smaller peg3 d321)
(smaller d107 d459)
(smaller d321 d459)
(smaller d321 d107)
(clear peg2)
(clear peg3)
(clear d459)
(on d321 peg1)
(on d107 d321)
(on d459 d107)
)
(:goal
(and 
(on d321 peg3)
(on d107 d321)
(on d459 d107)
)
)
)


