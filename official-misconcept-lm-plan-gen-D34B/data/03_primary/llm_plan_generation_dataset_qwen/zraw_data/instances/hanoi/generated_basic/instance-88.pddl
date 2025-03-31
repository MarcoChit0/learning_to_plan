


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d367 d289 )
(:init
(smaller peg1 d367)
(smaller peg1 d289)
(smaller peg2 d367)
(smaller peg2 d289)
(smaller peg3 d367)
(smaller peg3 d289)
(smaller d289 d367)
(clear peg2)
(clear peg3)
(clear d367)
(on d289 peg1)
(on d367 d289)
)
(:goal
(and 
(on d289 peg3)
(on d367 d289)
)
)
)


