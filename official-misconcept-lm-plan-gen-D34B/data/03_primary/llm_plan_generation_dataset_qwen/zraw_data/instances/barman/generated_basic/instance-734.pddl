(define (problem prob)
 (:domain barman)
 (:objects 
      shaker54 - shaker
      left right - hand
      shot475 - shot
      ingredient313 ingredient401 ingredient205 - ingredient
      cocktail1 - cocktail
      dispenser5 dispenser137 dispenser163 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker54)
  (ontable shot475)
  (dispenses dispenser5 ingredient313)
  (dispenses dispenser137 ingredient401)
  (dispenses dispenser163 ingredient205)
  (clean shaker54)
  (clean shot475)
  (empty shaker54)
  (empty shot475)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker54 l0)
  (shaker-level shaker54 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient205)
  (cocktail-part2 cocktail1 ingredient401)
)
 (:goal
  (and
      (contains shot475 cocktail1)
)))
