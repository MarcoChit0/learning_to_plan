(define (problem prob)
 (:domain barman)
 (:objects 
      shaker429 - shaker
      left right - hand
      shot481 - shot
      ingredient306 ingredient427 ingredient346 - ingredient
      cocktail1 - cocktail
      dispenser91 dispenser216 dispenser404 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker429)
  (ontable shot481)
  (dispenses dispenser91 ingredient306)
  (dispenses dispenser216 ingredient427)
  (dispenses dispenser404 ingredient346)
  (clean shaker429)
  (clean shot481)
  (empty shaker429)
  (empty shot481)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker429 l0)
  (shaker-level shaker429 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient346)
  (cocktail-part2 cocktail1 ingredient427)
)
 (:goal
  (and
      (contains shot481 cocktail1)
)))
