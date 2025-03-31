


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d207 d89 )
(:init
(smaller peg1 d207)
(smaller peg1 d89)
(smaller peg2 d207)
(smaller peg2 d89)
(smaller peg3 d207)
(smaller peg3 d89)
(smaller d89 d207)
(clear peg2)
(clear peg3)
(clear d207)
(on d89 peg1)
(on d207 d89)
)
(:goal
(and 
(on d89 peg3)
(on d207 d89)
)
)
)


