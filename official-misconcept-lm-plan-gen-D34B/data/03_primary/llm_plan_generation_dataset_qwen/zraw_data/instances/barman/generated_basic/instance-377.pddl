(define (problem prob)
 (:domain barman)
 (:objects 
      shaker3 - shaker
      left right - hand
      shot146 - shot
      ingredient0 ingredient474 - ingredient
      cocktail1 - cocktail
      dispenser393 dispenser424 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker3)
  (ontable shot146)
  (dispenses dispenser393 ingredient0)
  (dispenses dispenser424 ingredient474)
  (clean shaker3)
  (clean shot146)
  (empty shaker3)
  (empty shot146)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker3 l0)
  (shaker-level shaker3 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient474)
  (cocktail-part2 cocktail1 ingredient0)
)
 (:goal
  (and
      (contains shot146 cocktail1)
)))
