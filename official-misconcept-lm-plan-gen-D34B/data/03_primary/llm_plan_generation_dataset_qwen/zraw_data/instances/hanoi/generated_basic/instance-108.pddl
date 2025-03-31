


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d464 d288 d3 )
(:init
(smaller peg1 d464)
(smaller peg1 d288)
(smaller peg1 d3)
(smaller peg2 d464)
(smaller peg2 d288)
(smaller peg2 d3)
(smaller peg3 d464)
(smaller peg3 d288)
(smaller peg3 d3)
(smaller d288 d464)
(smaller d3 d464)
(smaller d3 d288)
(clear peg2)
(clear peg3)
(clear d464)
(on d3 peg1)
(on d288 d3)
(on d464 d288)
)
(:goal
(and 
(on d3 peg3)
(on d288 d3)
(on d464 d288)
)
)
)


