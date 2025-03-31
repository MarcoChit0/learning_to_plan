(define (problem prob)
 (:domain barman)
 (:objects 
      shaker458 - shaker
      left right - hand
      shot5 - shot
      ingredient403 ingredient436 - ingredient
      cocktail1 - cocktail
      dispenser119 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker458)
  (ontable shot5)
  (dispenses dispenser119 ingredient403)
  (dispenses dispenser64 ingredient436)
  (clean shaker458)
  (clean shot5)
  (empty shaker458)
  (empty shot5)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker458 l0)
  (shaker-level shaker458 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient436)
  (cocktail-part2 cocktail1 ingredient403)
)
 (:goal
  (and
      (contains shot5 cocktail1)
)))
