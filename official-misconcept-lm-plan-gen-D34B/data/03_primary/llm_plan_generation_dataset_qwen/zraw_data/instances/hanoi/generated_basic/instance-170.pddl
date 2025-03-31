


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d342 d317 d244 d453 )
(:init
(smaller peg1 d342)
(smaller peg1 d317)
(smaller peg1 d244)
(smaller peg1 d453)
(smaller peg2 d342)
(smaller peg2 d317)
(smaller peg2 d244)
(smaller peg2 d453)
(smaller peg3 d342)
(smaller peg3 d317)
(smaller peg3 d244)
(smaller peg3 d453)
(smaller d317 d342)
(smaller d244 d342)
(smaller d453 d342)
(smaller d244 d317)
(smaller d453 d317)
(smaller d453 d244)
(clear peg2)
(clear peg3)
(clear d342)
(on d453 peg1)
(on d244 d453)
(on d317 d244)
(on d342 d317)
)
(:goal
(and 
(on d453 peg3)
(on d244 d453)
(on d317 d244)
(on d342 d317)
)
)
)


