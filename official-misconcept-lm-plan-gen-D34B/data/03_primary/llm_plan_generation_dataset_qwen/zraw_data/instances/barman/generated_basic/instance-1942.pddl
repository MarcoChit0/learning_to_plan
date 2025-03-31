(define (problem prob)
 (:domain barman)
 (:objects 
      shaker347 - shaker
      left right - hand
      shot331 - shot
      ingredient278 ingredient8 - ingredient
      cocktail83 - cocktail
      dispenser265 dispenser3 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker347)
  (ontable shot331)
  (dispenses dispenser265 ingredient278)
  (dispenses dispenser3 ingredient8)
  (clean shaker347)
  (clean shot331)
  (empty shaker347)
  (empty shot331)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker347 l0)
  (shaker-level shaker347 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient278)
  (cocktail-part2 cocktail83 ingredient8)
)
 (:goal
  (and
      (contains shot331 cocktail83)
)))
