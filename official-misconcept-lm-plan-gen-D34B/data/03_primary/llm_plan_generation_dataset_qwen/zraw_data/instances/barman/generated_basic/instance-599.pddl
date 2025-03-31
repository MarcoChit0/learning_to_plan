(define (problem prob)
 (:domain barman)
 (:objects 
      shaker486 - shaker
      left right - hand
      shot477 - shot
      ingredient176 ingredient397 ingredient72 ingredient427 - ingredient
      cocktail1 - cocktail
      dispenser407 dispenser340 dispenser464 dispenser143 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker486)
  (ontable shot477)
  (dispenses dispenser407 ingredient176)
  (dispenses dispenser340 ingredient397)
  (dispenses dispenser464 ingredient72)
  (dispenses dispenser143 ingredient427)
  (clean shaker486)
  (clean shot477)
  (empty shaker486)
  (empty shot477)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker486 l0)
  (shaker-level shaker486 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient397)
  (cocktail-part2 cocktail1 ingredient427)
)
 (:goal
  (and
      (contains shot477 cocktail1)
)))
