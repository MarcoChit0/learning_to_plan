


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d133 d203 d495 d11 )
(:init
(smaller peg1 d133)
(smaller peg1 d203)
(smaller peg1 d495)
(smaller peg1 d11)
(smaller peg2 d133)
(smaller peg2 d203)
(smaller peg2 d495)
(smaller peg2 d11)
(smaller peg3 d133)
(smaller peg3 d203)
(smaller peg3 d495)
(smaller peg3 d11)
(smaller d203 d133)
(smaller d495 d133)
(smaller d11 d133)
(smaller d495 d203)
(smaller d11 d203)
(smaller d11 d495)
(clear peg2)
(clear peg3)
(clear d133)
(on d11 peg1)
(on d495 d11)
(on d203 d495)
(on d133 d203)
)
(:goal
(and 
(on d11 peg3)
(on d495 d11)
(on d203 d495)
(on d133 d203)
)
)
)


