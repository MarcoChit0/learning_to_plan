(define (problem prob)
 (:domain barman)
 (:objects 
      shaker484 - shaker
      left right - hand
      shot273 shot125 - shot
      ingredient477 ingredient351 ingredient202 - ingredient
      cocktail1 - cocktail
      dispenser222 dispenser116 dispenser358 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker484)
  (ontable shot273)
  (ontable shot125)
  (dispenses dispenser222 ingredient477)
  (dispenses dispenser116 ingredient351)
  (dispenses dispenser358 ingredient202)
  (clean shaker484)
  (clean shot273)
  (clean shot125)
  (empty shaker484)
  (empty shot273)
  (empty shot125)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker484 l0)
  (shaker-level shaker484 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient351)
  (cocktail-part2 cocktail1 ingredient202)
)
 (:goal
  (and
      (contains shot273 cocktail1)
)))
