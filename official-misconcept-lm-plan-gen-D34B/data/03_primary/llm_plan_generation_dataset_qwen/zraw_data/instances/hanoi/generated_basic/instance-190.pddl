


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d415 d472 d62 d329 )
(:init
(smaller peg1 d415)
(smaller peg1 d472)
(smaller peg1 d62)
(smaller peg1 d329)
(smaller peg2 d415)
(smaller peg2 d472)
(smaller peg2 d62)
(smaller peg2 d329)
(smaller peg3 d415)
(smaller peg3 d472)
(smaller peg3 d62)
(smaller peg3 d329)
(smaller d472 d415)
(smaller d62 d415)
(smaller d329 d415)
(smaller d62 d472)
(smaller d329 d472)
(smaller d329 d62)
(clear peg2)
(clear peg3)
(clear d415)
(on d329 peg1)
(on d62 d329)
(on d472 d62)
(on d415 d472)
)
(:goal
(and 
(on d329 peg3)
(on d62 d329)
(on d472 d62)
(on d415 d472)
)
)
)


