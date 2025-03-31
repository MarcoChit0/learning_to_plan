(define (problem prob)
 (:domain barman)
 (:objects 
      shaker486 - shaker
      left right - hand
      shot277 - shot
      ingredient55 ingredient287 - ingredient
      cocktail1 - cocktail
      dispenser488 dispenser381 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker486)
  (ontable shot277)
  (dispenses dispenser488 ingredient55)
  (dispenses dispenser381 ingredient287)
  (clean shaker486)
  (clean shot277)
  (empty shaker486)
  (empty shot277)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker486 l0)
  (shaker-level shaker486 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient287)
  (cocktail-part2 cocktail1 ingredient55)
)
 (:goal
  (and
      (contains shot277 cocktail1)
)))
