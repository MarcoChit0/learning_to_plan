


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d68 d41 d416 d118 )
(:init
(smaller peg1 d68)
(smaller peg1 d41)
(smaller peg1 d416)
(smaller peg1 d118)
(smaller peg2 d68)
(smaller peg2 d41)
(smaller peg2 d416)
(smaller peg2 d118)
(smaller peg3 d68)
(smaller peg3 d41)
(smaller peg3 d416)
(smaller peg3 d118)
(smaller d41 d68)
(smaller d416 d68)
(smaller d118 d68)
(smaller d416 d41)
(smaller d118 d41)
(smaller d118 d416)
(clear peg2)
(clear peg3)
(clear d68)
(on d118 peg1)
(on d416 d118)
(on d41 d416)
(on d68 d41)
)
(:goal
(and 
(on d118 peg3)
(on d416 d118)
(on d41 d416)
(on d68 d41)
)
)
)


