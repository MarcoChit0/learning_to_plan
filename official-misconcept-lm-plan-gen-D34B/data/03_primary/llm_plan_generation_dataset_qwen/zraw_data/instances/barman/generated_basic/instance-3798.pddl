(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot497 - shot
      ingredient360 ingredient165 ingredient390 ingredient275 - ingredient
      cocktail46 - cocktail
      dispenser442 dispenser41 dispenser235 dispenser224 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot497)
  (dispenses dispenser442 ingredient360)
  (dispenses dispenser41 ingredient165)
  (dispenses dispenser235 ingredient390)
  (dispenses dispenser224 ingredient275)
  (clean shaker167)
  (clean shot497)
  (empty shaker167)
  (empty shot497)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail46 ingredient165)
  (cocktail-part2 cocktail46 ingredient275)
)
 (:goal
  (and
      (contains shot497 cocktail46)
)))
