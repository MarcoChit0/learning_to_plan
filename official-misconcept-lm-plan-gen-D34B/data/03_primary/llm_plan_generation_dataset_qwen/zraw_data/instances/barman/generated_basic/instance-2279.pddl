(define (problem prob)
 (:domain barman)
 (:objects 
      shaker107 - shaker
      left right - hand
      shot102 shot451 - shot
      ingredient70 ingredient54 ingredient478 ingredient91 - ingredient
      cocktail165 - cocktail
      dispenser237 dispenser198 dispenser29 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker107)
  (ontable shot102)
  (ontable shot451)
  (dispenses dispenser237 ingredient70)
  (dispenses dispenser198 ingredient54)
  (dispenses dispenser29 ingredient478)
  (dispenses dispenser73 ingredient91)
  (clean shaker107)
  (clean shot102)
  (clean shot451)
  (empty shaker107)
  (empty shot102)
  (empty shot451)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker107 l0)
  (shaker-level shaker107 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail165 ingredient54)
  (cocktail-part2 cocktail165 ingredient70)
)
 (:goal
  (and
      (contains shot102 cocktail165)
)))
