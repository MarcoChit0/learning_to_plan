


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d149 d494 d482 d245 )
(:init
(smaller peg1 d149)
(smaller peg1 d494)
(smaller peg1 d482)
(smaller peg1 d245)
(smaller peg2 d149)
(smaller peg2 d494)
(smaller peg2 d482)
(smaller peg2 d245)
(smaller peg3 d149)
(smaller peg3 d494)
(smaller peg3 d482)
(smaller peg3 d245)
(smaller d494 d149)
(smaller d482 d149)
(smaller d245 d149)
(smaller d482 d494)
(smaller d245 d494)
(smaller d245 d482)
(clear peg2)
(clear peg3)
(clear d149)
(on d245 peg1)
(on d482 d245)
(on d494 d482)
(on d149 d494)
)
(:goal
(and 
(on d245 peg3)
(on d482 d245)
(on d494 d482)
(on d149 d494)
)
)
)


