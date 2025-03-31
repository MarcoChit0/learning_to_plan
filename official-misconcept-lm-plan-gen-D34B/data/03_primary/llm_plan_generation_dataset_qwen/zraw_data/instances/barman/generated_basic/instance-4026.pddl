(define (problem prob)
 (:domain barman)
 (:objects 
      shaker402 - shaker
      left right - hand
      shot234 shot241 - shot
      ingredient169 ingredient20 ingredient305 - ingredient
      cocktail46 - cocktail
      dispenser276 dispenser475 dispenser247 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker402)
  (ontable shot234)
  (ontable shot241)
  (dispenses dispenser276 ingredient169)
  (dispenses dispenser475 ingredient20)
  (dispenses dispenser247 ingredient305)
  (clean shaker402)
  (clean shot234)
  (clean shot241)
  (empty shaker402)
  (empty shot234)
  (empty shot241)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker402 l0)
  (shaker-level shaker402 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail46 ingredient20)
  (cocktail-part2 cocktail46 ingredient305)
)
 (:goal
  (and
      (contains shot234 cocktail46)
)))
