


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d425 d163 d245 d310 )
(:init
(smaller peg1 d425)
(smaller peg1 d163)
(smaller peg1 d245)
(smaller peg1 d310)
(smaller peg2 d425)
(smaller peg2 d163)
(smaller peg2 d245)
(smaller peg2 d310)
(smaller peg3 d425)
(smaller peg3 d163)
(smaller peg3 d245)
(smaller peg3 d310)
(smaller d163 d425)
(smaller d245 d425)
(smaller d310 d425)
(smaller d245 d163)
(smaller d310 d163)
(smaller d310 d245)
(clear peg2)
(clear peg3)
(clear d425)
(on d310 peg1)
(on d245 d310)
(on d163 d245)
(on d425 d163)
)
(:goal
(and 
(on d310 peg3)
(on d245 d310)
(on d163 d245)
(on d425 d163)
)
)
)


