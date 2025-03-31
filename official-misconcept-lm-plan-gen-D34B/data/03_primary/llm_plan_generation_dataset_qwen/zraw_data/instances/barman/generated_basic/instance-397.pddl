(define (problem prob)
 (:domain barman)
 (:objects 
      shaker213 - shaker
      left right - hand
      shot13 shot84 - shot
      ingredient474 ingredient42 - ingredient
      cocktail1 - cocktail
      dispenser68 dispenser465 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker213)
  (ontable shot13)
  (ontable shot84)
  (dispenses dispenser68 ingredient474)
  (dispenses dispenser465 ingredient42)
  (clean shaker213)
  (clean shot13)
  (clean shot84)
  (empty shaker213)
  (empty shot13)
  (empty shot84)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker213 l0)
  (shaker-level shaker213 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient42)
  (cocktail-part2 cocktail1 ingredient474)
)
 (:goal
  (and
      (contains shot13 cocktail1)
)))
