(define (problem prob)
 (:domain barman)
 (:objects 
      shaker378 - shaker
      left right - hand
      shot129 shot89 - shot
      ingredient234 ingredient113 ingredient13 - ingredient
      cocktail328 - cocktail
      dispenser378 dispenser144 dispenser125 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker378)
  (ontable shot129)
  (ontable shot89)
  (dispenses dispenser378 ingredient234)
  (dispenses dispenser144 ingredient113)
  (dispenses dispenser125 ingredient13)
  (clean shaker378)
  (clean shot129)
  (clean shot89)
  (empty shaker378)
  (empty shot129)
  (empty shot89)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker378 l0)
  (shaker-level shaker378 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail328 ingredient13)
  (cocktail-part2 cocktail328 ingredient113)
)
 (:goal
  (and
      (contains shot129 cocktail328)
)))
