


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d438 d125 d423 d33 )
(:init
(smaller peg1 d438)
(smaller peg1 d125)
(smaller peg1 d423)
(smaller peg1 d33)
(smaller peg2 d438)
(smaller peg2 d125)
(smaller peg2 d423)
(smaller peg2 d33)
(smaller peg3 d438)
(smaller peg3 d125)
(smaller peg3 d423)
(smaller peg3 d33)
(smaller d125 d438)
(smaller d423 d438)
(smaller d33 d438)
(smaller d423 d125)
(smaller d33 d125)
(smaller d33 d423)
(clear peg2)
(clear peg3)
(clear d438)
(on d33 peg1)
(on d423 d33)
(on d125 d423)
(on d438 d125)
)
(:goal
(and 
(on d33 peg3)
(on d423 d33)
(on d125 d423)
(on d438 d125)
)
)
)


