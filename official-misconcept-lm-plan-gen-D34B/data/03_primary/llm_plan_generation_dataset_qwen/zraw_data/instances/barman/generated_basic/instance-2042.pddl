(define (problem prob)
 (:domain barman)
 (:objects 
      shaker418 - shaker
      left right - hand
      shot491 - shot
      ingredient446 ingredient148 ingredient461 ingredient272 - ingredient
      cocktail294 - cocktail
      dispenser172 dispenser296 dispenser421 dispenser12 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker418)
  (ontable shot491)
  (dispenses dispenser172 ingredient446)
  (dispenses dispenser296 ingredient148)
  (dispenses dispenser421 ingredient461)
  (dispenses dispenser12 ingredient272)
  (clean shaker418)
  (clean shot491)
  (empty shaker418)
  (empty shot491)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker418 l0)
  (shaker-level shaker418 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail294 ingredient461)
  (cocktail-part2 cocktail294 ingredient446)
)
 (:goal
  (and
      (contains shot491 cocktail294)
)))
