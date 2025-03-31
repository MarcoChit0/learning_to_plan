(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot204 shot218 shot66 - shot
      ingredient216 ingredient310 ingredient179 ingredient247 - ingredient
      cocktail194 - cocktail
      dispenser392 dispenser193 dispenser5 dispenser232 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot204)
  (ontable shot218)
  (ontable shot66)
  (dispenses dispenser392 ingredient216)
  (dispenses dispenser193 ingredient310)
  (dispenses dispenser5 ingredient179)
  (dispenses dispenser232 ingredient247)
  (clean shaker426)
  (clean shot204)
  (clean shot218)
  (clean shot66)
  (empty shaker426)
  (empty shot204)
  (empty shot218)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail194 ingredient179)
  (cocktail-part2 cocktail194 ingredient310)
)
 (:goal
  (and
      (contains shot204 cocktail194)
      (contains shot218 ingredient179)
)))
