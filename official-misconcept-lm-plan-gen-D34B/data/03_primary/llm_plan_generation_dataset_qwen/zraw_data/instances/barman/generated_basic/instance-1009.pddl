(define (problem prob)
 (:domain barman)
 (:objects 
      shaker181 - shaker
      left right - hand
      shot347 - shot
      ingredient80 ingredient481 ingredient259 - ingredient
      cocktail1 - cocktail
      dispenser386 dispenser459 dispenser487 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker181)
  (ontable shot347)
  (dispenses dispenser386 ingredient80)
  (dispenses dispenser459 ingredient481)
  (dispenses dispenser487 ingredient259)
  (clean shaker181)
  (clean shot347)
  (empty shaker181)
  (empty shot347)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker181 l0)
  (shaker-level shaker181 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient481)
  (cocktail-part2 cocktail1 ingredient80)
)
 (:goal
  (and
      (contains shot347 cocktail1)
)))
