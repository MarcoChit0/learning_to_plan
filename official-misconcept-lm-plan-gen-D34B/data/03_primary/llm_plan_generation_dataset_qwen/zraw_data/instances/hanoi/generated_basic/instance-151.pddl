


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d18 d316 d125 d39 )
(:init
(smaller peg1 d18)
(smaller peg1 d316)
(smaller peg1 d125)
(smaller peg1 d39)
(smaller peg2 d18)
(smaller peg2 d316)
(smaller peg2 d125)
(smaller peg2 d39)
(smaller peg3 d18)
(smaller peg3 d316)
(smaller peg3 d125)
(smaller peg3 d39)
(smaller d316 d18)
(smaller d125 d18)
(smaller d39 d18)
(smaller d125 d316)
(smaller d39 d316)
(smaller d39 d125)
(clear peg2)
(clear peg3)
(clear d18)
(on d39 peg1)
(on d125 d39)
(on d316 d125)
(on d18 d316)
)
(:goal
(and 
(on d39 peg3)
(on d125 d39)
(on d316 d125)
(on d18 d316)
)
)
)


