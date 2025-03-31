(define (problem prob)
 (:domain barman)
 (:objects 
      shaker402 - shaker
      left right - hand
      shot467 shot208 shot411 - shot
      ingredient479 ingredient228 ingredient245 ingredient359 - ingredient
      cocktail1 - cocktail
      dispenser418 dispenser195 dispenser20 dispenser398 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker402)
  (ontable shot467)
  (ontable shot208)
  (ontable shot411)
  (dispenses dispenser418 ingredient479)
  (dispenses dispenser195 ingredient228)
  (dispenses dispenser20 ingredient245)
  (dispenses dispenser398 ingredient359)
  (clean shaker402)
  (clean shot467)
  (clean shot208)
  (clean shot411)
  (empty shaker402)
  (empty shot467)
  (empty shot208)
  (empty shot411)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker402 l0)
  (shaker-level shaker402 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient359)
  (cocktail-part2 cocktail1 ingredient228)
)
 (:goal
  (and
      (contains shot467 cocktail1)
      (contains shot208 ingredient245)
)))
