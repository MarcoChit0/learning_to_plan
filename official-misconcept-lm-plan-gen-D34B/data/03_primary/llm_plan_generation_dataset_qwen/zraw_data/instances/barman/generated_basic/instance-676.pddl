(define (problem prob)
 (:domain barman)
 (:objects 
      shaker214 - shaker
      left right - hand
      shot351 shot255 - shot
      ingredient121 ingredient457 - ingredient
      cocktail1 - cocktail
      dispenser70 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker214)
  (ontable shot351)
  (ontable shot255)
  (dispenses dispenser70 ingredient121)
  (dispenses dispenser214 ingredient457)
  (clean shaker214)
  (clean shot351)
  (clean shot255)
  (empty shaker214)
  (empty shot351)
  (empty shot255)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker214 l0)
  (shaker-level shaker214 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient457)
  (cocktail-part2 cocktail1 ingredient121)
)
 (:goal
  (and
      (contains shot351 cocktail1)
)))
