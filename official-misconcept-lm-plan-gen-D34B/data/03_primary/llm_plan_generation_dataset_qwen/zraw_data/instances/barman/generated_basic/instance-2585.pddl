(define (problem prob)
 (:domain barman)
 (:objects 
      shaker26 - shaker
      left right - hand
      shot34 - shot
      ingredient380 ingredient211 - ingredient
      cocktail14 - cocktail
      dispenser432 dispenser254 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker26)
  (ontable shot34)
  (dispenses dispenser432 ingredient380)
  (dispenses dispenser254 ingredient211)
  (clean shaker26)
  (clean shot34)
  (empty shaker26)
  (empty shot34)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker26 l0)
  (shaker-level shaker26 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail14 ingredient211)
  (cocktail-part2 cocktail14 ingredient380)
)
 (:goal
  (and
      (contains shot34 cocktail14)
)))
