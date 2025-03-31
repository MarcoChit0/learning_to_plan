(define (problem prob)
 (:domain barman)
 (:objects 
      shaker409 - shaker
      left right - hand
      shot406 - shot
      ingredient286 ingredient62 ingredient457 ingredient203 - ingredient
      cocktail453 - cocktail
      dispenser235 dispenser271 dispenser403 dispenser251 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker409)
  (ontable shot406)
  (dispenses dispenser235 ingredient286)
  (dispenses dispenser271 ingredient62)
  (dispenses dispenser403 ingredient457)
  (dispenses dispenser251 ingredient203)
  (clean shaker409)
  (clean shot406)
  (empty shaker409)
  (empty shot406)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker409 l0)
  (shaker-level shaker409 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail453 ingredient62)
  (cocktail-part2 cocktail453 ingredient203)
)
 (:goal
  (and
      (contains shot406 cocktail453)
)))
