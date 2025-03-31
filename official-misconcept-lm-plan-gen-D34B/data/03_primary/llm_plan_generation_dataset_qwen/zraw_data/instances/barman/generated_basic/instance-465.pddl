(define (problem prob)
 (:domain barman)
 (:objects 
      shaker85 - shaker
      left right - hand
      shot390 shot274 shot499 - shot
      ingredient376 ingredient178 ingredient206 ingredient357 - ingredient
      cocktail1 - cocktail
      dispenser16 dispenser362 dispenser351 dispenser345 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker85)
  (ontable shot390)
  (ontable shot274)
  (ontable shot499)
  (dispenses dispenser16 ingredient376)
  (dispenses dispenser362 ingredient178)
  (dispenses dispenser351 ingredient206)
  (dispenses dispenser345 ingredient357)
  (clean shaker85)
  (clean shot390)
  (clean shot274)
  (clean shot499)
  (empty shaker85)
  (empty shot390)
  (empty shot274)
  (empty shot499)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker85 l0)
  (shaker-level shaker85 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient206)
  (cocktail-part2 cocktail1 ingredient357)
)
 (:goal
  (and
      (contains shot390 cocktail1)
      (contains shot274 cocktail1)
)))
