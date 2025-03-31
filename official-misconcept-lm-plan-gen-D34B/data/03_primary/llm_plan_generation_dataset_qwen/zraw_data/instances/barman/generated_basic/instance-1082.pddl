(define (problem prob)
 (:domain barman)
 (:objects 
      shaker337 - shaker
      left right - hand
      shot471 shot31 shot345 - shot
      ingredient56 ingredient325 ingredient336 ingredient185 - ingredient
      cocktail1 - cocktail
      dispenser6 dispenser447 dispenser264 dispenser19 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker337)
  (ontable shot471)
  (ontable shot31)
  (ontable shot345)
  (dispenses dispenser6 ingredient56)
  (dispenses dispenser447 ingredient325)
  (dispenses dispenser264 ingredient336)
  (dispenses dispenser19 ingredient185)
  (clean shaker337)
  (clean shot471)
  (clean shot31)
  (clean shot345)
  (empty shaker337)
  (empty shot471)
  (empty shot31)
  (empty shot345)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker337 l0)
  (shaker-level shaker337 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient185)
  (cocktail-part2 cocktail1 ingredient56)
)
 (:goal
  (and
      (contains shot471 cocktail1)
      (contains shot31 cocktail1)
)))
