(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot124 shot474 shot470 - shot
      ingredient32 ingredient364 ingredient350 - ingredient
      cocktail6 - cocktail
      dispenser140 dispenser96 dispenser420 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot124)
  (ontable shot474)
  (ontable shot470)
  (dispenses dispenser140 ingredient32)
  (dispenses dispenser96 ingredient364)
  (dispenses dispenser420 ingredient350)
  (clean shaker294)
  (clean shot124)
  (clean shot474)
  (clean shot470)
  (empty shaker294)
  (empty shot124)
  (empty shot474)
  (empty shot470)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail6 ingredient350)
  (cocktail-part2 cocktail6 ingredient364)
)
 (:goal
  (and
      (contains shot124 cocktail6)
      (contains shot474 ingredient32)
)))
