(define (problem prob)
 (:domain barman)
 (:objects 
      shaker319 - shaker
      left right - hand
      shot116 shot226 shot209 - shot
      ingredient93 ingredient191 ingredient50 - ingredient
      cocktail72 - cocktail
      dispenser334 dispenser199 dispenser300 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker319)
  (ontable shot116)
  (ontable shot226)
  (ontable shot209)
  (dispenses dispenser334 ingredient93)
  (dispenses dispenser199 ingredient191)
  (dispenses dispenser300 ingredient50)
  (clean shaker319)
  (clean shot116)
  (clean shot226)
  (clean shot209)
  (empty shaker319)
  (empty shot116)
  (empty shot226)
  (empty shot209)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker319 l0)
  (shaker-level shaker319 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail72 ingredient50)
  (cocktail-part2 cocktail72 ingredient191)
)
 (:goal
  (and
      (contains shot116 cocktail72)
      (contains shot226 cocktail72)
)))
