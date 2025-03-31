


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d435 )
(:init
(smaller peg1 d435)
(smaller peg2 d435)
(smaller peg3 d435)
(clear peg2)
(clear peg3)
(clear d435)
(on d435 peg1)
)
(:goal
(and 
(on d435 peg3)
)
)
)


