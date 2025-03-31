


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d58 d204 d119 )
(:init
(smaller peg1 d58)
(smaller peg1 d204)
(smaller peg1 d119)
(smaller peg2 d58)
(smaller peg2 d204)
(smaller peg2 d119)
(smaller peg3 d58)
(smaller peg3 d204)
(smaller peg3 d119)
(smaller d204 d58)
(smaller d119 d58)
(smaller d119 d204)
(clear peg2)
(clear peg3)
(clear d58)
(on d119 peg1)
(on d204 d119)
(on d58 d204)
)
(:goal
(and 
(on d119 peg3)
(on d204 d119)
(on d58 d204)
)
)
)


