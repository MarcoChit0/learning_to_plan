


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d428 d321 )
(:init
(smaller peg1 d428)
(smaller peg1 d321)
(smaller peg2 d428)
(smaller peg2 d321)
(smaller peg3 d428)
(smaller peg3 d321)
(smaller d321 d428)
(clear peg2)
(clear peg3)
(clear d428)
(on d321 peg1)
(on d428 d321)
)
(:goal
(and 
(on d321 peg3)
(on d428 d321)
)
)
)


