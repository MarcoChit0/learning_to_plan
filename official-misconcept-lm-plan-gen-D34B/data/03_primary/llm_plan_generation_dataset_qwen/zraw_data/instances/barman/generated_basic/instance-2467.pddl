(define (problem prob)
 (:domain barman)
 (:objects 
      shaker31 - shaker
      left right - hand
      shot426 - shot
      ingredient294 ingredient320 ingredient491 - ingredient
      cocktail414 - cocktail
      dispenser274 dispenser238 dispenser373 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker31)
  (ontable shot426)
  (dispenses dispenser274 ingredient294)
  (dispenses dispenser238 ingredient320)
  (dispenses dispenser373 ingredient491)
  (clean shaker31)
  (clean shot426)
  (empty shaker31)
  (empty shot426)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker31 l0)
  (shaker-level shaker31 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail414 ingredient294)
  (cocktail-part2 cocktail414 ingredient491)
)
 (:goal
  (and
      (contains shot426 cocktail414)
)))
