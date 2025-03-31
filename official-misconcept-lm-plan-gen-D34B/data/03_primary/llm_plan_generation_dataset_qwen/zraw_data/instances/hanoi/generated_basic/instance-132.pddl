


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d353 d139 d208 )
(:init
(smaller peg1 d353)
(smaller peg1 d139)
(smaller peg1 d208)
(smaller peg2 d353)
(smaller peg2 d139)
(smaller peg2 d208)
(smaller peg3 d353)
(smaller peg3 d139)
(smaller peg3 d208)
(smaller d139 d353)
(smaller d208 d353)
(smaller d208 d139)
(clear peg2)
(clear peg3)
(clear d353)
(on d208 peg1)
(on d139 d208)
(on d353 d139)
)
(:goal
(and 
(on d208 peg3)
(on d139 d208)
(on d353 d139)
)
)
)


