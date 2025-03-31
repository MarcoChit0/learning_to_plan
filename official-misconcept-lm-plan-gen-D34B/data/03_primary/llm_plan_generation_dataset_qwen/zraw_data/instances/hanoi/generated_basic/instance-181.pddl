


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d131 d349 d138 d107 )
(:init
(smaller peg1 d131)
(smaller peg1 d349)
(smaller peg1 d138)
(smaller peg1 d107)
(smaller peg2 d131)
(smaller peg2 d349)
(smaller peg2 d138)
(smaller peg2 d107)
(smaller peg3 d131)
(smaller peg3 d349)
(smaller peg3 d138)
(smaller peg3 d107)
(smaller d349 d131)
(smaller d138 d131)
(smaller d107 d131)
(smaller d138 d349)
(smaller d107 d349)
(smaller d107 d138)
(clear peg2)
(clear peg3)
(clear d131)
(on d107 peg1)
(on d138 d107)
(on d349 d138)
(on d131 d349)
)
(:goal
(and 
(on d107 peg3)
(on d138 d107)
(on d349 d138)
(on d131 d349)
)
)
)


