(define (problem prob)
 (:domain barman)
 (:objects 
      shaker49 - shaker
      left right - hand
      shot449 shot240 - shot
      ingredient232 ingredient121 ingredient492 ingredient123 - ingredient
      cocktail1 - cocktail
      dispenser271 dispenser408 dispenser358 dispenser20 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker49)
  (ontable shot449)
  (ontable shot240)
  (dispenses dispenser271 ingredient232)
  (dispenses dispenser408 ingredient121)
  (dispenses dispenser358 ingredient492)
  (dispenses dispenser20 ingredient123)
  (clean shaker49)
  (clean shot449)
  (clean shot240)
  (empty shaker49)
  (empty shot449)
  (empty shot240)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker49 l0)
  (shaker-level shaker49 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient123)
  (cocktail-part2 cocktail1 ingredient121)
)
 (:goal
  (and
      (contains shot449 cocktail1)
)))
