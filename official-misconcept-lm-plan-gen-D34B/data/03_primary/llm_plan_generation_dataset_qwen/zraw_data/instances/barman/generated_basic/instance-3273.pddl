(define (problem prob)
 (:domain barman)
 (:objects 
      shaker342 - shaker
      left right - hand
      shot472 shot276 - shot
      ingredient323 ingredient21 ingredient480 - ingredient
      cocktail294 - cocktail
      dispenser43 dispenser216 dispenser295 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker342)
  (ontable shot472)
  (ontable shot276)
  (dispenses dispenser43 ingredient323)
  (dispenses dispenser216 ingredient21)
  (dispenses dispenser295 ingredient480)
  (clean shaker342)
  (clean shot472)
  (clean shot276)
  (empty shaker342)
  (empty shot472)
  (empty shot276)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker342 l0)
  (shaker-level shaker342 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail294 ingredient323)
  (cocktail-part2 cocktail294 ingredient21)
)
 (:goal
  (and
      (contains shot472 cocktail294)
)))
