


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d262 d478 d176 d103 )
(:init
(smaller peg1 d262)
(smaller peg1 d478)
(smaller peg1 d176)
(smaller peg1 d103)
(smaller peg2 d262)
(smaller peg2 d478)
(smaller peg2 d176)
(smaller peg2 d103)
(smaller peg3 d262)
(smaller peg3 d478)
(smaller peg3 d176)
(smaller peg3 d103)
(smaller d478 d262)
(smaller d176 d262)
(smaller d103 d262)
(smaller d176 d478)
(smaller d103 d478)
(smaller d103 d176)
(clear peg2)
(clear peg3)
(clear d262)
(on d103 peg1)
(on d176 d103)
(on d478 d176)
(on d262 d478)
)
(:goal
(and 
(on d103 peg3)
(on d176 d103)
(on d478 d176)
(on d262 d478)
)
)
)


