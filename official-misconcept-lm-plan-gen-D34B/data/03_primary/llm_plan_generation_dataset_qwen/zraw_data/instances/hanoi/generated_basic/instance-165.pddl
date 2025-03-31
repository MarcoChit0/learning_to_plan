


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d200 d169 d204 d213 )
(:init
(smaller peg1 d200)
(smaller peg1 d169)
(smaller peg1 d204)
(smaller peg1 d213)
(smaller peg2 d200)
(smaller peg2 d169)
(smaller peg2 d204)
(smaller peg2 d213)
(smaller peg3 d200)
(smaller peg3 d169)
(smaller peg3 d204)
(smaller peg3 d213)
(smaller d169 d200)
(smaller d204 d200)
(smaller d213 d200)
(smaller d204 d169)
(smaller d213 d169)
(smaller d213 d204)
(clear peg2)
(clear peg3)
(clear d200)
(on d213 peg1)
(on d204 d213)
(on d169 d204)
(on d200 d169)
)
(:goal
(and 
(on d213 peg3)
(on d204 d213)
(on d169 d204)
(on d200 d169)
)
)
)


