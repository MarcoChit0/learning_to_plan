(define (problem prob)
 (:domain barman)
 (:objects 
      shaker414 - shaker
      left right - hand
      shot81 shot148 shot490 - shot
      ingredient447 ingredient260 - ingredient
      cocktail37 - cocktail
      dispenser214 dispenser209 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker414)
  (ontable shot81)
  (ontable shot148)
  (ontable shot490)
  (dispenses dispenser214 ingredient447)
  (dispenses dispenser209 ingredient260)
  (clean shaker414)
  (clean shot81)
  (clean shot148)
  (clean shot490)
  (empty shaker414)
  (empty shot81)
  (empty shot148)
  (empty shot490)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker414 l0)
  (shaker-level shaker414 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail37 ingredient447)
  (cocktail-part2 cocktail37 ingredient260)
)
 (:goal
  (and
      (contains shot81 cocktail37)
      (contains shot148 cocktail37)
)))
