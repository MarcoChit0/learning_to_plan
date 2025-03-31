(define (problem prob)
 (:domain barman)
 (:objects 
      shaker123 - shaker
      left right - hand
      shot273 - shot
      ingredient189 ingredient31 - ingredient
      cocktail168 - cocktail
      dispenser47 dispenser248 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker123)
  (ontable shot273)
  (dispenses dispenser47 ingredient189)
  (dispenses dispenser248 ingredient31)
  (clean shaker123)
  (clean shot273)
  (empty shaker123)
  (empty shot273)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker123 l0)
  (shaker-level shaker123 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail168 ingredient31)
  (cocktail-part2 cocktail168 ingredient189)
)
 (:goal
  (and
      (contains shot273 cocktail168)
)))
