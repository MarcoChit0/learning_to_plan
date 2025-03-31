(define (problem prob)
 (:domain barman)
 (:objects 
      shaker35 - shaker
      left right - hand
      shot137 - shot
      ingredient392 ingredient475 ingredient372 ingredient359 - ingredient
      cocktail207 - cocktail
      dispenser123 dispenser193 dispenser100 dispenser71 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker35)
  (ontable shot137)
  (dispenses dispenser123 ingredient392)
  (dispenses dispenser193 ingredient475)
  (dispenses dispenser100 ingredient372)
  (dispenses dispenser71 ingredient359)
  (clean shaker35)
  (clean shot137)
  (empty shaker35)
  (empty shot137)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker35 l0)
  (shaker-level shaker35 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail207 ingredient475)
  (cocktail-part2 cocktail207 ingredient372)
)
 (:goal
  (and
      (contains shot137 cocktail207)
)))
