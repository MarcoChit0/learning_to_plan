(define (problem prob)
 (:domain barman)
 (:objects 
      shaker429 - shaker
      left right - hand
      shot95 - shot
      ingredient369 ingredient441 ingredient45 - ingredient
      cocktail1 - cocktail
      dispenser248 dispenser113 dispenser0 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker429)
  (ontable shot95)
  (dispenses dispenser248 ingredient369)
  (dispenses dispenser113 ingredient441)
  (dispenses dispenser0 ingredient45)
  (clean shaker429)
  (clean shot95)
  (empty shaker429)
  (empty shot95)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker429 l0)
  (shaker-level shaker429 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient45)
  (cocktail-part2 cocktail1 ingredient441)
)
 (:goal
  (and
      (contains shot95 cocktail1)
)))
