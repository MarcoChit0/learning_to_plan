


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d189 d172 d491 d386 )
(:init
(smaller peg1 d189)
(smaller peg1 d172)
(smaller peg1 d491)
(smaller peg1 d386)
(smaller peg2 d189)
(smaller peg2 d172)
(smaller peg2 d491)
(smaller peg2 d386)
(smaller peg3 d189)
(smaller peg3 d172)
(smaller peg3 d491)
(smaller peg3 d386)
(smaller d172 d189)
(smaller d491 d189)
(smaller d386 d189)
(smaller d491 d172)
(smaller d386 d172)
(smaller d386 d491)
(clear peg2)
(clear peg3)
(clear d189)
(on d386 peg1)
(on d491 d386)
(on d172 d491)
(on d189 d172)
)
(:goal
(and 
(on d386 peg3)
(on d491 d386)
(on d172 d491)
(on d189 d172)
)
)
)


