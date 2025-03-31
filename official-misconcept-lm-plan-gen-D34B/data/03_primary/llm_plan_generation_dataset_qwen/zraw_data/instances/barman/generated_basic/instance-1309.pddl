(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot175 shot31 shot497 - shot
      ingredient224 ingredient204 - ingredient
      cocktail205 - cocktail
      dispenser260 dispenser406 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot175)
  (ontable shot31)
  (ontable shot497)
  (dispenses dispenser260 ingredient224)
  (dispenses dispenser406 ingredient204)
  (clean shaker468)
  (clean shot175)
  (clean shot31)
  (clean shot497)
  (empty shaker468)
  (empty shot175)
  (empty shot31)
  (empty shot497)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail205 ingredient224)
  (cocktail-part2 cocktail205 ingredient204)
)
 (:goal
  (and
      (contains shot175 cocktail205)
      (contains shot31 ingredient204)
)))
