(define (problem prob)
 (:domain barman)
 (:objects 
      shaker481 - shaker
      left right - hand
      shot412 shot403 - shot
      ingredient309 ingredient174 - ingredient
      cocktail203 - cocktail
      dispenser430 dispenser18 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker481)
  (ontable shot412)
  (ontable shot403)
  (dispenses dispenser430 ingredient309)
  (dispenses dispenser18 ingredient174)
  (clean shaker481)
  (clean shot412)
  (clean shot403)
  (empty shaker481)
  (empty shot412)
  (empty shot403)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker481 l0)
  (shaker-level shaker481 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail203 ingredient309)
  (cocktail-part2 cocktail203 ingredient174)
)
 (:goal
  (and
      (contains shot412 cocktail203)
)))
