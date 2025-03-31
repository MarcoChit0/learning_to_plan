


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d125 d180 d44 d319 )
(:init
(smaller peg1 d125)
(smaller peg1 d180)
(smaller peg1 d44)
(smaller peg1 d319)
(smaller peg2 d125)
(smaller peg2 d180)
(smaller peg2 d44)
(smaller peg2 d319)
(smaller peg3 d125)
(smaller peg3 d180)
(smaller peg3 d44)
(smaller peg3 d319)
(smaller d180 d125)
(smaller d44 d125)
(smaller d319 d125)
(smaller d44 d180)
(smaller d319 d180)
(smaller d319 d44)
(clear peg2)
(clear peg3)
(clear d125)
(on d319 peg1)
(on d44 d319)
(on d180 d44)
(on d125 d180)
)
(:goal
(and 
(on d319 peg3)
(on d44 d319)
(on d180 d44)
(on d125 d180)
)
)
)


