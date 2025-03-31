(define (problem prob)
 (:domain barman)
 (:objects 
      shaker480 - shaker
      left right - hand
      shot401 - shot
      ingredient154 ingredient23 ingredient256 ingredient215 - ingredient
      cocktail89 - cocktail
      dispenser351 dispenser430 dispenser428 dispenser279 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker480)
  (ontable shot401)
  (dispenses dispenser351 ingredient154)
  (dispenses dispenser430 ingredient23)
  (dispenses dispenser428 ingredient256)
  (dispenses dispenser279 ingredient215)
  (clean shaker480)
  (clean shot401)
  (empty shaker480)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker480 l0)
  (shaker-level shaker480 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail89 ingredient215)
  (cocktail-part2 cocktail89 ingredient256)
)
 (:goal
  (and
      (contains shot401 cocktail89)
)))
