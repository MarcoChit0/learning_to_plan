


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d168 d160 d408 d273 )
(:init
(smaller peg1 d168)
(smaller peg1 d160)
(smaller peg1 d408)
(smaller peg1 d273)
(smaller peg2 d168)
(smaller peg2 d160)
(smaller peg2 d408)
(smaller peg2 d273)
(smaller peg3 d168)
(smaller peg3 d160)
(smaller peg3 d408)
(smaller peg3 d273)
(smaller d160 d168)
(smaller d408 d168)
(smaller d273 d168)
(smaller d408 d160)
(smaller d273 d160)
(smaller d273 d408)
(clear peg2)
(clear peg3)
(clear d168)
(on d273 peg1)
(on d408 d273)
(on d160 d408)
(on d168 d160)
)
(:goal
(and 
(on d273 peg3)
(on d408 d273)
(on d160 d408)
(on d168 d160)
)
)
)


