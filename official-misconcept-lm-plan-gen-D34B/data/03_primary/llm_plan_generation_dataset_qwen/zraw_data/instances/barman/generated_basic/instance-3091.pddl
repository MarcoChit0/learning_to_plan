(define (problem prob)
 (:domain barman)
 (:objects 
      shaker238 - shaker
      left right - hand
      shot437 - shot
      ingredient165 ingredient123 ingredient167 ingredient366 - ingredient
      cocktail351 - cocktail
      dispenser294 dispenser280 dispenser468 dispenser95 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker238)
  (ontable shot437)
  (dispenses dispenser294 ingredient165)
  (dispenses dispenser280 ingredient123)
  (dispenses dispenser468 ingredient167)
  (dispenses dispenser95 ingredient366)
  (clean shaker238)
  (clean shot437)
  (empty shaker238)
  (empty shot437)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker238 l0)
  (shaker-level shaker238 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail351 ingredient165)
  (cocktail-part2 cocktail351 ingredient167)
)
 (:goal
  (and
      (contains shot437 cocktail351)
)))
