(define (problem prob)
 (:domain barman)
 (:objects 
      shaker297 - shaker
      left right - hand
      shot38 - shot
      ingredient140 ingredient490 ingredient247 - ingredient
      cocktail1 - cocktail
      dispenser476 dispenser481 dispenser209 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker297)
  (ontable shot38)
  (dispenses dispenser476 ingredient140)
  (dispenses dispenser481 ingredient490)
  (dispenses dispenser209 ingredient247)
  (clean shaker297)
  (clean shot38)
  (empty shaker297)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker297 l0)
  (shaker-level shaker297 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient247)
  (cocktail-part2 cocktail1 ingredient490)
)
 (:goal
  (and
      (contains shot38 cocktail1)
)))
