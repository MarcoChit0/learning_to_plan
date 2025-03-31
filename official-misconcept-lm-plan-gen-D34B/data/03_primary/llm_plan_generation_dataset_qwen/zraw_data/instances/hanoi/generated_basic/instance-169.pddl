


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d453 d107 d61 d62 )
(:init
(smaller peg1 d453)
(smaller peg1 d107)
(smaller peg1 d61)
(smaller peg1 d62)
(smaller peg2 d453)
(smaller peg2 d107)
(smaller peg2 d61)
(smaller peg2 d62)
(smaller peg3 d453)
(smaller peg3 d107)
(smaller peg3 d61)
(smaller peg3 d62)
(smaller d107 d453)
(smaller d61 d453)
(smaller d62 d453)
(smaller d61 d107)
(smaller d62 d107)
(smaller d62 d61)
(clear peg2)
(clear peg3)
(clear d453)
(on d62 peg1)
(on d61 d62)
(on d107 d61)
(on d453 d107)
)
(:goal
(and 
(on d62 peg3)
(on d61 d62)
(on d107 d61)
(on d453 d107)
)
)
)


