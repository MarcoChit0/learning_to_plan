(define (problem prob)
 (:domain barman)
 (:objects 
      shaker58 - shaker
      left right - hand
      shot358 shot309 shot123 - shot
      ingredient351 ingredient360 ingredient457 - ingredient
      cocktail159 - cocktail
      dispenser216 dispenser425 dispenser286 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker58)
  (ontable shot358)
  (ontable shot309)
  (ontable shot123)
  (dispenses dispenser216 ingredient351)
  (dispenses dispenser425 ingredient360)
  (dispenses dispenser286 ingredient457)
  (clean shaker58)
  (clean shot358)
  (clean shot309)
  (clean shot123)
  (empty shaker58)
  (empty shot358)
  (empty shot309)
  (empty shot123)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker58 l0)
  (shaker-level shaker58 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail159 ingredient457)
  (cocktail-part2 cocktail159 ingredient351)
)
 (:goal
  (and
      (contains shot358 cocktail159)
      (contains shot309 cocktail159)
)))
