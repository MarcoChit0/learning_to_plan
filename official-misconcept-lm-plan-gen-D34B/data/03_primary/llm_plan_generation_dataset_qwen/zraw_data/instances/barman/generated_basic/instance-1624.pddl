(define (problem prob)
 (:domain barman)
 (:objects 
      shaker224 - shaker
      left right - hand
      shot469 shot5 shot255 - shot
      ingredient163 ingredient87 - ingredient
      cocktail130 - cocktail
      dispenser100 dispenser192 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker224)
  (ontable shot469)
  (ontable shot5)
  (ontable shot255)
  (dispenses dispenser100 ingredient163)
  (dispenses dispenser192 ingredient87)
  (clean shaker224)
  (clean shot469)
  (clean shot5)
  (clean shot255)
  (empty shaker224)
  (empty shot469)
  (empty shot5)
  (empty shot255)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker224 l0)
  (shaker-level shaker224 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail130 ingredient163)
  (cocktail-part2 cocktail130 ingredient87)
)
 (:goal
  (and
      (contains shot469 cocktail130)
      (contains shot5 cocktail130)
)))
