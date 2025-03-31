(define (problem prob)
 (:domain barman)
 (:objects 
      shaker395 - shaker
      left right - hand
      shot376 shot62 - shot
      ingredient47 ingredient496 - ingredient
      cocktail1 - cocktail
      dispenser486 dispenser493 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker395)
  (ontable shot376)
  (ontable shot62)
  (dispenses dispenser486 ingredient47)
  (dispenses dispenser493 ingredient496)
  (clean shaker395)
  (clean shot376)
  (clean shot62)
  (empty shaker395)
  (empty shot376)
  (empty shot62)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker395 l0)
  (shaker-level shaker395 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient47)
  (cocktail-part2 cocktail1 ingredient496)
)
 (:goal
  (and
      (contains shot376 cocktail1)
)))
