


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d163 d42 d372 )
(:init
(smaller peg1 d163)
(smaller peg1 d42)
(smaller peg1 d372)
(smaller peg2 d163)
(smaller peg2 d42)
(smaller peg2 d372)
(smaller peg3 d163)
(smaller peg3 d42)
(smaller peg3 d372)
(smaller d42 d163)
(smaller d372 d163)
(smaller d372 d42)
(clear peg2)
(clear peg3)
(clear d163)
(on d372 peg1)
(on d42 d372)
(on d163 d42)
)
(:goal
(and 
(on d372 peg3)
(on d42 d372)
(on d163 d42)
)
)
)


