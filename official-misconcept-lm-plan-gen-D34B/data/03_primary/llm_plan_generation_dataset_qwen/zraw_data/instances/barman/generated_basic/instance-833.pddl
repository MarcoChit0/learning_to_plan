(define (problem prob)
 (:domain barman)
 (:objects 
      shaker31 - shaker
      left right - hand
      shot382 shot245 shot149 - shot
      ingredient394 ingredient499 ingredient2 ingredient477 - ingredient
      cocktail1 - cocktail
      dispenser130 dispenser230 dispenser151 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker31)
  (ontable shot382)
  (ontable shot245)
  (ontable shot149)
  (dispenses dispenser130 ingredient394)
  (dispenses dispenser230 ingredient499)
  (dispenses dispenser151 ingredient2)
  (dispenses dispenser458 ingredient477)
  (clean shaker31)
  (clean shot382)
  (clean shot245)
  (clean shot149)
  (empty shaker31)
  (empty shot382)
  (empty shot245)
  (empty shot149)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker31 l0)
  (shaker-level shaker31 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient499)
  (cocktail-part2 cocktail1 ingredient477)
)
 (:goal
  (and
      (contains shot382 cocktail1)
      (contains shot245 cocktail1)
)))
