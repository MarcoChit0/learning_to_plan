


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d241 d231 d193 d460 )
(:init
(smaller peg1 d241)
(smaller peg1 d231)
(smaller peg1 d193)
(smaller peg1 d460)
(smaller peg2 d241)
(smaller peg2 d231)
(smaller peg2 d193)
(smaller peg2 d460)
(smaller peg3 d241)
(smaller peg3 d231)
(smaller peg3 d193)
(smaller peg3 d460)
(smaller d231 d241)
(smaller d193 d241)
(smaller d460 d241)
(smaller d193 d231)
(smaller d460 d231)
(smaller d460 d193)
(clear peg2)
(clear peg3)
(clear d241)
(on d460 peg1)
(on d193 d460)
(on d231 d193)
(on d241 d231)
)
(:goal
(and 
(on d460 peg3)
(on d193 d460)
(on d231 d193)
(on d241 d231)
)
)
)


