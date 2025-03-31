


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d321 d72 d20 )
(:init
(smaller peg1 d321)
(smaller peg1 d72)
(smaller peg1 d20)
(smaller peg2 d321)
(smaller peg2 d72)
(smaller peg2 d20)
(smaller peg3 d321)
(smaller peg3 d72)
(smaller peg3 d20)
(smaller d72 d321)
(smaller d20 d321)
(smaller d20 d72)
(clear peg2)
(clear peg3)
(clear d321)
(on d20 peg1)
(on d72 d20)
(on d321 d72)
)
(:goal
(and 
(on d20 peg3)
(on d72 d20)
(on d321 d72)
)
)
)


