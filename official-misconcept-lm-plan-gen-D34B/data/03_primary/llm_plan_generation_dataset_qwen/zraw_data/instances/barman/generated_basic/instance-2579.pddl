(define (problem prob)
 (:domain barman)
 (:objects 
      shaker33 - shaker
      left right - hand
      shot214 - shot
      ingredient266 ingredient42 - ingredient
      cocktail432 - cocktail
      dispenser437 dispenser136 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker33)
  (ontable shot214)
  (dispenses dispenser437 ingredient266)
  (dispenses dispenser136 ingredient42)
  (clean shaker33)
  (clean shot214)
  (empty shaker33)
  (empty shot214)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker33 l0)
  (shaker-level shaker33 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail432 ingredient266)
  (cocktail-part2 cocktail432 ingredient42)
)
 (:goal
  (and
      (contains shot214 cocktail432)
)))
