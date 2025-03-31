(define (problem prob)
 (:domain barman)
 (:objects 
      shaker134 - shaker
      left right - hand
      shot49 shot380 shot455 - shot
      ingredient416 ingredient335 - ingredient
      cocktail24 - cocktail
      dispenser117 dispenser59 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker134)
  (ontable shot49)
  (ontable shot380)
  (ontable shot455)
  (dispenses dispenser117 ingredient416)
  (dispenses dispenser59 ingredient335)
  (clean shaker134)
  (clean shot49)
  (clean shot380)
  (clean shot455)
  (empty shaker134)
  (empty shot49)
  (empty shot380)
  (empty shot455)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker134 l0)
  (shaker-level shaker134 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail24 ingredient416)
  (cocktail-part2 cocktail24 ingredient335)
)
 (:goal
  (and
      (contains shot49 cocktail24)
      (contains shot380 cocktail24)
)))
