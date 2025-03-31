


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d472 d169 d482 )
(:init
(smaller peg1 d472)
(smaller peg1 d169)
(smaller peg1 d482)
(smaller peg2 d472)
(smaller peg2 d169)
(smaller peg2 d482)
(smaller peg3 d472)
(smaller peg3 d169)
(smaller peg3 d482)
(smaller d169 d472)
(smaller d482 d472)
(smaller d482 d169)
(clear peg2)
(clear peg3)
(clear d472)
(on d482 peg1)
(on d169 d482)
(on d472 d169)
)
(:goal
(and 
(on d482 peg3)
(on d169 d482)
(on d472 d169)
)
)
)


