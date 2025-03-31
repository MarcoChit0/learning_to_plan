(define (problem prob)
 (:domain barman)
 (:objects 
      shaker325 - shaker
      left right - hand
      shot117 shot186 - shot
      ingredient489 ingredient224 ingredient465 ingredient331 - ingredient
      cocktail1 - cocktail
      dispenser129 dispenser392 dispenser9 dispenser282 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker325)
  (ontable shot117)
  (ontable shot186)
  (dispenses dispenser129 ingredient489)
  (dispenses dispenser392 ingredient224)
  (dispenses dispenser9 ingredient465)
  (dispenses dispenser282 ingredient331)
  (clean shaker325)
  (clean shot117)
  (clean shot186)
  (empty shaker325)
  (empty shot117)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker325 l0)
  (shaker-level shaker325 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient465)
  (cocktail-part2 cocktail1 ingredient489)
)
 (:goal
  (and
      (contains shot117 cocktail1)
)))
