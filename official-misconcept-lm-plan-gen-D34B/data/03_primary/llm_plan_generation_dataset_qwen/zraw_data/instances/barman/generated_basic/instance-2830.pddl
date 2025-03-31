(define (problem prob)
 (:domain barman)
 (:objects 
      shaker108 - shaker
      left right - hand
      shot491 shot468 shot47 - shot
      ingredient72 ingredient286 ingredient450 ingredient148 - ingredient
      cocktail408 - cocktail
      dispenser300 dispenser487 dispenser325 dispenser486 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker108)
  (ontable shot491)
  (ontable shot468)
  (ontable shot47)
  (dispenses dispenser300 ingredient72)
  (dispenses dispenser487 ingredient286)
  (dispenses dispenser325 ingredient450)
  (dispenses dispenser486 ingredient148)
  (clean shaker108)
  (clean shot491)
  (clean shot468)
  (clean shot47)
  (empty shaker108)
  (empty shot491)
  (empty shot468)
  (empty shot47)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker108 l0)
  (shaker-level shaker108 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail408 ingredient72)
  (cocktail-part2 cocktail408 ingredient286)
)
 (:goal
  (and
      (contains shot491 cocktail408)
      (contains shot468 ingredient286)
)))
