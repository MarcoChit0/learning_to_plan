(define (problem prob)
 (:domain barman)
 (:objects 
      shaker22 - shaker
      left right - hand
      shot148 shot14 shot256 - shot
      ingredient448 ingredient257 ingredient49 ingredient358 - ingredient
      cocktail54 - cocktail
      dispenser107 dispenser374 dispenser397 dispenser428 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker22)
  (ontable shot148)
  (ontable shot14)
  (ontable shot256)
  (dispenses dispenser107 ingredient448)
  (dispenses dispenser374 ingredient257)
  (dispenses dispenser397 ingredient49)
  (dispenses dispenser428 ingredient358)
  (clean shaker22)
  (clean shot148)
  (clean shot14)
  (clean shot256)
  (empty shaker22)
  (empty shot148)
  (empty shot14)
  (empty shot256)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker22 l0)
  (shaker-level shaker22 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail54 ingredient49)
  (cocktail-part2 cocktail54 ingredient257)
)
 (:goal
  (and
      (contains shot148 cocktail54)
      (contains shot14 cocktail54)
)))
