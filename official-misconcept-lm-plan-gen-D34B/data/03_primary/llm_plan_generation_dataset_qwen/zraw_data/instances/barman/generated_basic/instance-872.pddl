(define (problem prob)
 (:domain barman)
 (:objects 
      shaker51 - shaker
      left right - hand
      shot326 - shot
      ingredient95 ingredient453 - ingredient
      cocktail1 - cocktail
      dispenser398 dispenser434 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker51)
  (ontable shot326)
  (dispenses dispenser398 ingredient95)
  (dispenses dispenser434 ingredient453)
  (clean shaker51)
  (clean shot326)
  (empty shaker51)
  (empty shot326)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker51 l0)
  (shaker-level shaker51 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient95)
  (cocktail-part2 cocktail1 ingredient453)
)
 (:goal
  (and
      (contains shot326 cocktail1)
)))
