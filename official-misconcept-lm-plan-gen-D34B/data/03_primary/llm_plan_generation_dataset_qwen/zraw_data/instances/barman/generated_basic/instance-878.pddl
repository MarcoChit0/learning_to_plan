(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot403 - shot
      ingredient479 ingredient333 - ingredient
      cocktail1 - cocktail
      dispenser68 dispenser197 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot403)
  (dispenses dispenser68 ingredient479)
  (dispenses dispenser197 ingredient333)
  (clean shaker469)
  (clean shot403)
  (empty shaker469)
  (empty shot403)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient479)
  (cocktail-part2 cocktail1 ingredient333)
)
 (:goal
  (and
      (contains shot403 cocktail1)
)))
