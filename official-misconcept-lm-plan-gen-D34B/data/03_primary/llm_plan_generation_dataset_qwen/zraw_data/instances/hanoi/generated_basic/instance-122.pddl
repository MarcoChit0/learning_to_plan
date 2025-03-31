


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d224 d318 d157 )
(:init
(smaller peg1 d224)
(smaller peg1 d318)
(smaller peg1 d157)
(smaller peg2 d224)
(smaller peg2 d318)
(smaller peg2 d157)
(smaller peg3 d224)
(smaller peg3 d318)
(smaller peg3 d157)
(smaller d318 d224)
(smaller d157 d224)
(smaller d157 d318)
(clear peg2)
(clear peg3)
(clear d224)
(on d157 peg1)
(on d318 d157)
(on d224 d318)
)
(:goal
(and 
(on d157 peg3)
(on d318 d157)
(on d224 d318)
)
)
)


