(define (problem prob)
 (:domain barman)
 (:objects 
      shaker341 - shaker
      left right - hand
      shot46 shot389 shot108 - shot
      ingredient406 ingredient33 ingredient313 ingredient273 - ingredient
      cocktail453 - cocktail
      dispenser427 dispenser368 dispenser210 dispenser203 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker341)
  (ontable shot46)
  (ontable shot389)
  (ontable shot108)
  (dispenses dispenser427 ingredient406)
  (dispenses dispenser368 ingredient33)
  (dispenses dispenser210 ingredient313)
  (dispenses dispenser203 ingredient273)
  (clean shaker341)
  (clean shot46)
  (clean shot389)
  (clean shot108)
  (empty shaker341)
  (empty shot46)
  (empty shot389)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker341 l0)
  (shaker-level shaker341 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail453 ingredient406)
  (cocktail-part2 cocktail453 ingredient313)
)
 (:goal
  (and
      (contains shot46 cocktail453)
      (contains shot389 cocktail453)
)))
