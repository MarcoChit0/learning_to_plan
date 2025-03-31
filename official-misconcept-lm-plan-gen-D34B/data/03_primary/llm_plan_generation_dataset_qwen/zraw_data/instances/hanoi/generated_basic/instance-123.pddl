


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d389 d224 d6 )
(:init
(smaller peg1 d389)
(smaller peg1 d224)
(smaller peg1 d6)
(smaller peg2 d389)
(smaller peg2 d224)
(smaller peg2 d6)
(smaller peg3 d389)
(smaller peg3 d224)
(smaller peg3 d6)
(smaller d224 d389)
(smaller d6 d389)
(smaller d6 d224)
(clear peg2)
(clear peg3)
(clear d389)
(on d6 peg1)
(on d224 d6)
(on d389 d224)
)
(:goal
(and 
(on d6 peg3)
(on d224 d6)
(on d389 d224)
)
)
)


