


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d405 d373 d190 )
(:init
(smaller peg1 d405)
(smaller peg1 d373)
(smaller peg1 d190)
(smaller peg2 d405)
(smaller peg2 d373)
(smaller peg2 d190)
(smaller peg3 d405)
(smaller peg3 d373)
(smaller peg3 d190)
(smaller d373 d405)
(smaller d190 d405)
(smaller d190 d373)
(clear peg2)
(clear peg3)
(clear d405)
(on d190 peg1)
(on d373 d190)
(on d405 d373)
)
(:goal
(and 
(on d190 peg3)
(on d373 d190)
(on d405 d373)
)
)
)


