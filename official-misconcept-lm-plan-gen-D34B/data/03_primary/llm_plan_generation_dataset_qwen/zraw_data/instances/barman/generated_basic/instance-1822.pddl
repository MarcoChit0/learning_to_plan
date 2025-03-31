(define (problem prob)
 (:domain barman)
 (:objects 
      shaker74 - shaker
      left right - hand
      shot297 shot321 shot335 - shot
      ingredient45 ingredient110 ingredient492 ingredient203 - ingredient
      cocktail81 - cocktail
      dispenser83 dispenser192 dispenser352 dispenser432 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker74)
  (ontable shot297)
  (ontable shot321)
  (ontable shot335)
  (dispenses dispenser83 ingredient45)
  (dispenses dispenser192 ingredient110)
  (dispenses dispenser352 ingredient492)
  (dispenses dispenser432 ingredient203)
  (clean shaker74)
  (clean shot297)
  (clean shot321)
  (clean shot335)
  (empty shaker74)
  (empty shot297)
  (empty shot321)
  (empty shot335)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker74 l0)
  (shaker-level shaker74 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail81 ingredient45)
  (cocktail-part2 cocktail81 ingredient110)
)
 (:goal
  (and
      (contains shot297 cocktail81)
      (contains shot321 cocktail81)
)))
