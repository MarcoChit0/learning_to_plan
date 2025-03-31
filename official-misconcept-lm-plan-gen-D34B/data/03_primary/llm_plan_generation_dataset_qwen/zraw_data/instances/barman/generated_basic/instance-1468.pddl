(define (problem prob)
 (:domain barman)
 (:objects 
      shaker19 - shaker
      left right - hand
      shot469 shot157 shot383 - shot
      ingredient347 ingredient119 - ingredient
      cocktail83 - cocktail
      dispenser276 dispenser107 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker19)
  (ontable shot469)
  (ontable shot157)
  (ontable shot383)
  (dispenses dispenser276 ingredient347)
  (dispenses dispenser107 ingredient119)
  (clean shaker19)
  (clean shot469)
  (clean shot157)
  (clean shot383)
  (empty shaker19)
  (empty shot469)
  (empty shot157)
  (empty shot383)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker19 l0)
  (shaker-level shaker19 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient119)
  (cocktail-part2 cocktail83 ingredient347)
)
 (:goal
  (and
      (contains shot469 cocktail83)
      (contains shot157 ingredient119)
)))
