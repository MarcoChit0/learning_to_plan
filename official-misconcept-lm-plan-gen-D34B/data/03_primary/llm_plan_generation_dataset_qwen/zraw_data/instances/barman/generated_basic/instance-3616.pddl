(define (problem prob)
 (:domain barman)
 (:objects 
      shaker278 - shaker
      left right - hand
      shot141 shot403 shot70 - shot
      ingredient172 ingredient27 ingredient132 - ingredient
      cocktail312 - cocktail
      dispenser237 dispenser475 dispenser373 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker278)
  (ontable shot141)
  (ontable shot403)
  (ontable shot70)
  (dispenses dispenser237 ingredient172)
  (dispenses dispenser475 ingredient27)
  (dispenses dispenser373 ingredient132)
  (clean shaker278)
  (clean shot141)
  (clean shot403)
  (clean shot70)
  (empty shaker278)
  (empty shot141)
  (empty shot403)
  (empty shot70)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker278 l0)
  (shaker-level shaker278 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail312 ingredient132)
  (cocktail-part2 cocktail312 ingredient27)
)
 (:goal
  (and
      (contains shot141 cocktail312)
      (contains shot403 ingredient27)
)))
