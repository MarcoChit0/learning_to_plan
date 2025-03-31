


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d112 d230 d235 )
(:init
(smaller peg1 d112)
(smaller peg1 d230)
(smaller peg1 d235)
(smaller peg2 d112)
(smaller peg2 d230)
(smaller peg2 d235)
(smaller peg3 d112)
(smaller peg3 d230)
(smaller peg3 d235)
(smaller d230 d112)
(smaller d235 d112)
(smaller d235 d230)
(clear peg2)
(clear peg3)
(clear d112)
(on d235 peg1)
(on d230 d235)
(on d112 d230)
)
(:goal
(and 
(on d235 peg3)
(on d230 d235)
(on d112 d230)
)
)
)


