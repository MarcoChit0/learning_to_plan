


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d339 d322 )
(:init
(smaller peg1 d339)
(smaller peg1 d322)
(smaller peg2 d339)
(smaller peg2 d322)
(smaller peg3 d339)
(smaller peg3 d322)
(smaller d322 d339)
(clear peg2)
(clear peg3)
(clear d339)
(on d322 peg1)
(on d339 d322)
)
(:goal
(and 
(on d322 peg3)
(on d339 d322)
)
)
)


