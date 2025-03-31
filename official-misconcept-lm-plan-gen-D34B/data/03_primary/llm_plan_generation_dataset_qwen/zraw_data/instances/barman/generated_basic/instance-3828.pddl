(define (problem prob)
 (:domain barman)
 (:objects 
      shaker118 - shaker
      left right - hand
      shot117 shot95 - shot
      ingredient101 ingredient193 ingredient62 - ingredient
      cocktail267 - cocktail
      dispenser171 dispenser72 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker118)
  (ontable shot117)
  (ontable shot95)
  (dispenses dispenser171 ingredient101)
  (dispenses dispenser72 ingredient193)
  (dispenses dispenser385 ingredient62)
  (clean shaker118)
  (clean shot117)
  (clean shot95)
  (empty shaker118)
  (empty shot117)
  (empty shot95)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker118 l0)
  (shaker-level shaker118 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail267 ingredient101)
  (cocktail-part2 cocktail267 ingredient62)
)
 (:goal
  (and
      (contains shot117 cocktail267)
)))
