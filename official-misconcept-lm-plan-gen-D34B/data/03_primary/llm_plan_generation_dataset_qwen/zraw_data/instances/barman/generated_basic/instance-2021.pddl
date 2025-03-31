(define (problem prob)
 (:domain barman)
 (:objects 
      shaker451 - shaker
      left right - hand
      shot448 shot169 - shot
      ingredient56 ingredient404 - ingredient
      cocktail69 - cocktail
      dispenser497 dispenser121 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker451)
  (ontable shot448)
  (ontable shot169)
  (dispenses dispenser497 ingredient56)
  (dispenses dispenser121 ingredient404)
  (clean shaker451)
  (clean shot448)
  (clean shot169)
  (empty shaker451)
  (empty shot448)
  (empty shot169)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker451 l0)
  (shaker-level shaker451 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail69 ingredient56)
  (cocktail-part2 cocktail69 ingredient404)
)
 (:goal
  (and
      (contains shot448 cocktail69)
)))
