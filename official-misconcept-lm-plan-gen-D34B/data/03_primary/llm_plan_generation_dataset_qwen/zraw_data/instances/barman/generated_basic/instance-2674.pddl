(define (problem prob)
 (:domain barman)
 (:objects 
      shaker3 - shaker
      left right - hand
      shot479 shot466 shot379 shot358 - shot
      ingredient217 ingredient23 - ingredient
      cocktail69 - cocktail
      dispenser316 dispenser216 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker3)
  (ontable shot479)
  (ontable shot466)
  (ontable shot379)
  (ontable shot358)
  (dispenses dispenser316 ingredient217)
  (dispenses dispenser216 ingredient23)
  (clean shaker3)
  (clean shot479)
  (clean shot466)
  (clean shot379)
  (clean shot358)
  (empty shaker3)
  (empty shot479)
  (empty shot466)
  (empty shot379)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker3 l0)
  (shaker-level shaker3 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail69 ingredient217)
  (cocktail-part2 cocktail69 ingredient23)
)
 (:goal
  (and
      (contains shot479 cocktail69)
      (contains shot466 cocktail69)
      (contains shot379 ingredient23)
)))
