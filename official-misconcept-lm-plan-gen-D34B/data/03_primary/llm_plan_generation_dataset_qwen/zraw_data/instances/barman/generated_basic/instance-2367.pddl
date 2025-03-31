(define (problem prob)
 (:domain barman)
 (:objects 
      shaker130 - shaker
      left right - hand
      shot399 - shot
      ingredient78 ingredient353 ingredient80 ingredient498 - ingredient
      cocktail328 - cocktail
      dispenser44 dispenser172 dispenser253 dispenser205 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker130)
  (ontable shot399)
  (dispenses dispenser44 ingredient78)
  (dispenses dispenser172 ingredient353)
  (dispenses dispenser253 ingredient80)
  (dispenses dispenser205 ingredient498)
  (clean shaker130)
  (clean shot399)
  (empty shaker130)
  (empty shot399)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker130 l0)
  (shaker-level shaker130 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail328 ingredient80)
  (cocktail-part2 cocktail328 ingredient498)
)
 (:goal
  (and
      (contains shot399 cocktail328)
)))
