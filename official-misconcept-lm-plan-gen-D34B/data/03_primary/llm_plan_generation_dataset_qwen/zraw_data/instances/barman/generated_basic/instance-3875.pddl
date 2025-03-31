(define (problem prob)
 (:domain barman)
 (:objects 
      shaker436 - shaker
      left right - hand
      shot87 shot381 - shot
      ingredient387 ingredient297 ingredient440 - ingredient
      cocktail373 - cocktail
      dispenser123 dispenser145 dispenser316 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker436)
  (ontable shot87)
  (ontable shot381)
  (dispenses dispenser123 ingredient387)
  (dispenses dispenser145 ingredient297)
  (dispenses dispenser316 ingredient440)
  (clean shaker436)
  (clean shot87)
  (clean shot381)
  (empty shaker436)
  (empty shot87)
  (empty shot381)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker436 l0)
  (shaker-level shaker436 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail373 ingredient297)
  (cocktail-part2 cocktail373 ingredient440)
)
 (:goal
  (and
      (contains shot87 cocktail373)
)))
