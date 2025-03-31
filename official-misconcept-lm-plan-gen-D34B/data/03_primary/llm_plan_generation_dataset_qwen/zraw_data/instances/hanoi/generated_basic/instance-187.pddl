


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d222 d399 d229 d67 )
(:init
(smaller peg1 d222)
(smaller peg1 d399)
(smaller peg1 d229)
(smaller peg1 d67)
(smaller peg2 d222)
(smaller peg2 d399)
(smaller peg2 d229)
(smaller peg2 d67)
(smaller peg3 d222)
(smaller peg3 d399)
(smaller peg3 d229)
(smaller peg3 d67)
(smaller d399 d222)
(smaller d229 d222)
(smaller d67 d222)
(smaller d229 d399)
(smaller d67 d399)
(smaller d67 d229)
(clear peg2)
(clear peg3)
(clear d222)
(on d67 peg1)
(on d229 d67)
(on d399 d229)
(on d222 d399)
)
(:goal
(and 
(on d67 peg3)
(on d229 d67)
(on d399 d229)
(on d222 d399)
)
)
)


