(define (problem prob)
 (:domain barman)
 (:objects 
      shaker87 - shaker
      left right - hand
      shot12 - shot
      ingredient407 ingredient179 - ingredient
      cocktail370 - cocktail
      dispenser128 dispenser121 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker87)
  (ontable shot12)
  (dispenses dispenser128 ingredient407)
  (dispenses dispenser121 ingredient179)
  (clean shaker87)
  (clean shot12)
  (empty shaker87)
  (empty shot12)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker87 l0)
  (shaker-level shaker87 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail370 ingredient407)
  (cocktail-part2 cocktail370 ingredient179)
)
 (:goal
  (and
      (contains shot12 cocktail370)
)))
