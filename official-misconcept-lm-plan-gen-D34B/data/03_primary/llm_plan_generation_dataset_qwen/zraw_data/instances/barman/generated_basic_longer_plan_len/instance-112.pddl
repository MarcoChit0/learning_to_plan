(define (problem prob)
 (:domain barman)
 (:objects 
      shaker329 - shaker
      left right - hand
      shot196 shot457 shot133 - shot
      ingredient366 ingredient390 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser401 dispenser329 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker329)
  (ontable shot196)
  (ontable shot457)
  (ontable shot133)
  (dispenses dispenser401 ingredient366)
  (dispenses dispenser329 ingredient390)
  (clean shaker329)
  (clean shot196)
  (clean shot457)
  (clean shot133)
  (empty shaker329)
  (empty shot196)
  (empty shot457)
  (empty shot133)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker329 l0)
  (shaker-level shaker329 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient390)
  (cocktail-part2 cocktail1 ingredient366)
  (cocktail-part1 cocktail2 ingredient366)
  (cocktail-part2 cocktail2 ingredient390)
)
 (:goal
  (and
      (contains shot196 cocktail1)
      (contains shot457 cocktail2)
)))
