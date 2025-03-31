(define (problem prob)
 (:domain barman)
 (:objects 
      shaker326 - shaker
      left right - hand
      shot438 shot290 shot357 shot383 - shot
      ingredient148 ingredient495 - ingredient
      cocktail1 - cocktail
      dispenser389 dispenser56 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker326)
  (ontable shot438)
  (ontable shot290)
  (ontable shot357)
  (ontable shot383)
  (dispenses dispenser389 ingredient148)
  (dispenses dispenser56 ingredient495)
  (clean shaker326)
  (clean shot438)
  (clean shot290)
  (clean shot357)
  (clean shot383)
  (empty shaker326)
  (empty shot438)
  (empty shot290)
  (empty shot357)
  (empty shot383)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker326 l0)
  (shaker-level shaker326 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient495)
  (cocktail-part2 cocktail1 ingredient148)
)
 (:goal
  (and
      (contains shot438 cocktail1)
      (contains shot290 ingredient495)
      (contains shot357 ingredient495)
)))
