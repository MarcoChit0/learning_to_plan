(define (problem prob)
 (:domain barman)
 (:objects 
      shaker96 - shaker
      left right - hand
      shot453 shot458 - shot
      ingredient411 ingredient163 - ingredient
      cocktail270 - cocktail
      dispenser91 dispenser463 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker96)
  (ontable shot453)
  (ontable shot458)
  (dispenses dispenser91 ingredient411)
  (dispenses dispenser463 ingredient163)
  (clean shaker96)
  (clean shot453)
  (clean shot458)
  (empty shaker96)
  (empty shot453)
  (empty shot458)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker96 l0)
  (shaker-level shaker96 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail270 ingredient163)
  (cocktail-part2 cocktail270 ingredient411)
)
 (:goal
  (and
      (contains shot453 cocktail270)
)))
