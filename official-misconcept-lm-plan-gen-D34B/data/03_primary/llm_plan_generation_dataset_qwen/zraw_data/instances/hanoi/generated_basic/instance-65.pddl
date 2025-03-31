


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d405 d232 )
(:init
(smaller peg1 d405)
(smaller peg1 d232)
(smaller peg2 d405)
(smaller peg2 d232)
(smaller peg3 d405)
(smaller peg3 d232)
(smaller d232 d405)
(clear peg2)
(clear peg3)
(clear d405)
(on d232 peg1)
(on d405 d232)
)
(:goal
(and 
(on d232 peg3)
(on d405 d232)
)
)
)


