(define (problem prob)
 (:domain barman)
 (:objects 
      shaker268 - shaker
      left right - hand
      shot488 shot96 shot339 - shot
      ingredient1 ingredient294 ingredient401 ingredient464 - ingredient
      cocktail54 - cocktail
      dispenser364 dispenser487 dispenser334 dispenser383 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker268)
  (ontable shot488)
  (ontable shot96)
  (ontable shot339)
  (dispenses dispenser364 ingredient1)
  (dispenses dispenser487 ingredient294)
  (dispenses dispenser334 ingredient401)
  (dispenses dispenser383 ingredient464)
  (clean shaker268)
  (clean shot488)
  (clean shot96)
  (clean shot339)
  (empty shaker268)
  (empty shot488)
  (empty shot96)
  (empty shot339)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker268 l0)
  (shaker-level shaker268 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail54 ingredient401)
  (cocktail-part2 cocktail54 ingredient294)
)
 (:goal
  (and
      (contains shot488 cocktail54)
      (contains shot96 cocktail54)
)))
