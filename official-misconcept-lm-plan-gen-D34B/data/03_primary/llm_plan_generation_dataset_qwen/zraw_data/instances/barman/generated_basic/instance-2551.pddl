(define (problem prob)
 (:domain barman)
 (:objects 
      shaker405 - shaker
      left right - hand
      shot391 shot403 shot94 - shot
      ingredient395 ingredient150 - ingredient
      cocktail403 - cocktail
      dispenser451 dispenser391 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker405)
  (ontable shot391)
  (ontable shot403)
  (ontable shot94)
  (dispenses dispenser451 ingredient395)
  (dispenses dispenser391 ingredient150)
  (clean shaker405)
  (clean shot391)
  (clean shot403)
  (clean shot94)
  (empty shaker405)
  (empty shot391)
  (empty shot403)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker405 l0)
  (shaker-level shaker405 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail403 ingredient395)
  (cocktail-part2 cocktail403 ingredient150)
)
 (:goal
  (and
      (contains shot391 cocktail403)
      (contains shot403 ingredient395)
)))
