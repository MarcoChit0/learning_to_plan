(define (problem prob)
 (:domain barman)
 (:objects 
      shaker117 - shaker
      left right - hand
      shot488 - shot
      ingredient467 ingredient388 ingredient12 - ingredient
      cocktail215 - cocktail
      dispenser427 dispenser83 dispenser146 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker117)
  (ontable shot488)
  (dispenses dispenser427 ingredient467)
  (dispenses dispenser83 ingredient388)
  (dispenses dispenser146 ingredient12)
  (clean shaker117)
  (clean shot488)
  (empty shaker117)
  (empty shot488)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker117 l0)
  (shaker-level shaker117 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail215 ingredient467)
  (cocktail-part2 cocktail215 ingredient12)
)
 (:goal
  (and
      (contains shot488 cocktail215)
)))
