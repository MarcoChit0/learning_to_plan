(define (problem prob)
 (:domain barman)
 (:objects 
      shaker3 - shaker
      left right - hand
      shot38 - shot
      ingredient110 ingredient359 - ingredient
      cocktail450 - cocktail
      dispenser82 dispenser198 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker3)
  (ontable shot38)
  (dispenses dispenser82 ingredient110)
  (dispenses dispenser198 ingredient359)
  (clean shaker3)
  (clean shot38)
  (empty shaker3)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker3 l0)
  (shaker-level shaker3 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail450 ingredient110)
  (cocktail-part2 cocktail450 ingredient359)
)
 (:goal
  (and
      (contains shot38 cocktail450)
)))
