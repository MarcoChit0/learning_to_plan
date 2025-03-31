(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot424 - shot
      ingredient486 ingredient292 - ingredient
      cocktail450 - cocktail
      dispenser270 dispenser21 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot424)
  (dispenses dispenser270 ingredient486)
  (dispenses dispenser21 ingredient292)
  (clean shaker294)
  (clean shot424)
  (empty shaker294)
  (empty shot424)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail450 ingredient486)
  (cocktail-part2 cocktail450 ingredient292)
)
 (:goal
  (and
      (contains shot424 cocktail450)
)))
