


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d330 d439 d137 )
(:init
(smaller peg1 d330)
(smaller peg1 d439)
(smaller peg1 d137)
(smaller peg2 d330)
(smaller peg2 d439)
(smaller peg2 d137)
(smaller peg3 d330)
(smaller peg3 d439)
(smaller peg3 d137)
(smaller d439 d330)
(smaller d137 d330)
(smaller d137 d439)
(clear peg2)
(clear peg3)
(clear d330)
(on d137 peg1)
(on d439 d137)
(on d330 d439)
)
(:goal
(and 
(on d137 peg3)
(on d439 d137)
(on d330 d439)
)
)
)


