(define (problem prob)
 (:domain barman)
 (:objects 
      shaker409 - shaker
      left right - hand
      shot351 - shot
      ingredient481 ingredient496 ingredient407 ingredient455 - ingredient
      cocktail1 - cocktail
      dispenser449 dispenser367 dispenser152 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker409)
  (ontable shot351)
  (dispenses dispenser449 ingredient481)
  (dispenses dispenser367 ingredient496)
  (dispenses dispenser152 ingredient407)
  (dispenses dispenser162 ingredient455)
  (clean shaker409)
  (clean shot351)
  (empty shaker409)
  (empty shot351)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker409 l0)
  (shaker-level shaker409 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient455)
  (cocktail-part2 cocktail1 ingredient407)
)
 (:goal
  (and
      (contains shot351 cocktail1)
)))
