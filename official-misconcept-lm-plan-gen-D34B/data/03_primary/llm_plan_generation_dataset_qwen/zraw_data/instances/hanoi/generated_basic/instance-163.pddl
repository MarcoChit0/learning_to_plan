


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d119 d159 d56 d117 )
(:init
(smaller peg1 d119)
(smaller peg1 d159)
(smaller peg1 d56)
(smaller peg1 d117)
(smaller peg2 d119)
(smaller peg2 d159)
(smaller peg2 d56)
(smaller peg2 d117)
(smaller peg3 d119)
(smaller peg3 d159)
(smaller peg3 d56)
(smaller peg3 d117)
(smaller d159 d119)
(smaller d56 d119)
(smaller d117 d119)
(smaller d56 d159)
(smaller d117 d159)
(smaller d117 d56)
(clear peg2)
(clear peg3)
(clear d119)
(on d117 peg1)
(on d56 d117)
(on d159 d56)
(on d119 d159)
)
(:goal
(and 
(on d117 peg3)
(on d56 d117)
(on d159 d56)
(on d119 d159)
)
)
)


