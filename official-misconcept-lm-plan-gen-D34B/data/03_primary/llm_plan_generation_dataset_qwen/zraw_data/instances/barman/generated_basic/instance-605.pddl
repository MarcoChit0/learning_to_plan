(define (problem prob)
 (:domain barman)
 (:objects 
      shaker365 - shaker
      left right - hand
      shot297 - shot
      ingredient325 ingredient128 ingredient453 ingredient279 - ingredient
      cocktail1 - cocktail
      dispenser299 dispenser460 dispenser46 dispenser453 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker365)
  (ontable shot297)
  (dispenses dispenser299 ingredient325)
  (dispenses dispenser460 ingredient128)
  (dispenses dispenser46 ingredient453)
  (dispenses dispenser453 ingredient279)
  (clean shaker365)
  (clean shot297)
  (empty shaker365)
  (empty shot297)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker365 l0)
  (shaker-level shaker365 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient279)
  (cocktail-part2 cocktail1 ingredient325)
)
 (:goal
  (and
      (contains shot297 cocktail1)
)))
