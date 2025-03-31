(define (problem prob)
 (:domain barman)
 (:objects 
      shaker310 - shaker
      left right - hand
      shot393 - shot
      ingredient499 ingredient146 ingredient486 - ingredient
      cocktail457 - cocktail
      dispenser485 dispenser47 dispenser58 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker310)
  (ontable shot393)
  (dispenses dispenser485 ingredient499)
  (dispenses dispenser47 ingredient146)
  (dispenses dispenser58 ingredient486)
  (clean shaker310)
  (clean shot393)
  (empty shaker310)
  (empty shot393)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker310 l0)
  (shaker-level shaker310 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail457 ingredient146)
  (cocktail-part2 cocktail457 ingredient486)
)
 (:goal
  (and
      (contains shot393 cocktail457)
)))
