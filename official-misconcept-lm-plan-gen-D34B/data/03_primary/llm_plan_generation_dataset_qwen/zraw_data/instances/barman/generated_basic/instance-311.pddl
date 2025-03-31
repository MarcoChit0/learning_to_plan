(define (problem prob)
 (:domain barman)
 (:objects 
      shaker196 - shaker
      left right - hand
      shot269 shot62 - shot
      ingredient1 ingredient430 - ingredient
      cocktail1 - cocktail
      dispenser368 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker196)
  (ontable shot269)
  (ontable shot62)
  (dispenses dispenser368 ingredient1)
  (dispenses dispenser64 ingredient430)
  (clean shaker196)
  (clean shot269)
  (clean shot62)
  (empty shaker196)
  (empty shot269)
  (empty shot62)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker196 l0)
  (shaker-level shaker196 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient430)
  (cocktail-part2 cocktail1 ingredient1)
)
 (:goal
  (and
      (contains shot269 cocktail1)
)))
