


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d454 d267 d275 d202 )
(:init
(smaller peg1 d454)
(smaller peg1 d267)
(smaller peg1 d275)
(smaller peg1 d202)
(smaller peg2 d454)
(smaller peg2 d267)
(smaller peg2 d275)
(smaller peg2 d202)
(smaller peg3 d454)
(smaller peg3 d267)
(smaller peg3 d275)
(smaller peg3 d202)
(smaller d267 d454)
(smaller d275 d454)
(smaller d202 d454)
(smaller d275 d267)
(smaller d202 d267)
(smaller d202 d275)
(clear peg2)
(clear peg3)
(clear d454)
(on d202 peg1)
(on d275 d202)
(on d267 d275)
(on d454 d267)
)
(:goal
(and 
(on d202 peg3)
(on d275 d202)
(on d267 d275)
(on d454 d267)
)
)
)


