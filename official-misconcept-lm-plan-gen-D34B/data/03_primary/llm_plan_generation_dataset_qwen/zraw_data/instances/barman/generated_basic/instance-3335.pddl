(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot457 shot475 - shot
      ingredient203 ingredient370 - ingredient
      cocktail135 - cocktail
      dispenser405 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot457)
  (ontable shot475)
  (dispenses dispenser405 ingredient203)
  (dispenses dispenser431 ingredient370)
  (clean shaker426)
  (clean shot457)
  (clean shot475)
  (empty shaker426)
  (empty shot457)
  (empty shot475)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail135 ingredient203)
  (cocktail-part2 cocktail135 ingredient370)
)
 (:goal
  (and
      (contains shot457 cocktail135)
)))
