(define (problem prob)
 (:domain barman)
 (:objects 
      shaker378 - shaker
      left right - hand
      shot403 shot281 shot284 - shot
      ingredient94 ingredient266 - ingredient
      cocktail405 - cocktail
      dispenser56 dispenser205 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker378)
  (ontable shot403)
  (ontable shot281)
  (ontable shot284)
  (dispenses dispenser56 ingredient94)
  (dispenses dispenser205 ingredient266)
  (clean shaker378)
  (clean shot403)
  (clean shot281)
  (clean shot284)
  (empty shaker378)
  (empty shot403)
  (empty shot281)
  (empty shot284)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker378 l0)
  (shaker-level shaker378 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail405 ingredient94)
  (cocktail-part2 cocktail405 ingredient266)
)
 (:goal
  (and
      (contains shot403 cocktail405)
      (contains shot281 ingredient94)
)))
