(define (problem prob)
 (:domain barman)
 (:objects 
      shaker50 - shaker
      left right - hand
      shot175 shot268 - shot
      ingredient389 ingredient313 ingredient403 ingredient492 - ingredient
      cocktail363 - cocktail
      dispenser78 dispenser243 dispenser473 dispenser275 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker50)
  (ontable shot175)
  (ontable shot268)
  (dispenses dispenser78 ingredient389)
  (dispenses dispenser243 ingredient313)
  (dispenses dispenser473 ingredient403)
  (dispenses dispenser275 ingredient492)
  (clean shaker50)
  (clean shot175)
  (clean shot268)
  (empty shaker50)
  (empty shot175)
  (empty shot268)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker50 l0)
  (shaker-level shaker50 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail363 ingredient389)
  (cocktail-part2 cocktail363 ingredient492)
)
 (:goal
  (and
      (contains shot175 cocktail363)
)))
