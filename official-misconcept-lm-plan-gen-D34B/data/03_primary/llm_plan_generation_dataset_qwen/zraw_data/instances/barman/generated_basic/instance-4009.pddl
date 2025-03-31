(define (problem prob)
 (:domain barman)
 (:objects 
      shaker373 - shaker
      left right - hand
      shot3 shot2 shot186 - shot
      ingredient35 ingredient432 - ingredient
      cocktail480 - cocktail
      dispenser292 dispenser185 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker373)
  (ontable shot3)
  (ontable shot2)
  (ontable shot186)
  (dispenses dispenser292 ingredient35)
  (dispenses dispenser185 ingredient432)
  (clean shaker373)
  (clean shot3)
  (clean shot2)
  (clean shot186)
  (empty shaker373)
  (empty shot3)
  (empty shot2)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker373 l0)
  (shaker-level shaker373 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail480 ingredient35)
  (cocktail-part2 cocktail480 ingredient432)
)
 (:goal
  (and
      (contains shot3 cocktail480)
      (contains shot2 cocktail480)
)))
