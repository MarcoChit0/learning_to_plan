


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d363 d63 )
(:init
(smaller peg1 d363)
(smaller peg1 d63)
(smaller peg2 d363)
(smaller peg2 d63)
(smaller peg3 d363)
(smaller peg3 d63)
(smaller d63 d363)
(clear peg2)
(clear peg3)
(clear d363)
(on d63 peg1)
(on d363 d63)
)
(:goal
(and 
(on d63 peg3)
(on d363 d63)
)
)
)


