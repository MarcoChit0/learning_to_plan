(define (problem prob)
 (:domain barman)
 (:objects 
      shaker163 - shaker
      left right - hand
      shot129 shot104 - shot
      ingredient200 ingredient170 - ingredient
      cocktail475 - cocktail
      dispenser61 dispenser264 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker163)
  (ontable shot129)
  (ontable shot104)
  (dispenses dispenser61 ingredient200)
  (dispenses dispenser264 ingredient170)
  (clean shaker163)
  (clean shot129)
  (clean shot104)
  (empty shaker163)
  (empty shot129)
  (empty shot104)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker163 l0)
  (shaker-level shaker163 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail475 ingredient170)
  (cocktail-part2 cocktail475 ingredient200)
)
 (:goal
  (and
      (contains shot129 cocktail475)
)))
