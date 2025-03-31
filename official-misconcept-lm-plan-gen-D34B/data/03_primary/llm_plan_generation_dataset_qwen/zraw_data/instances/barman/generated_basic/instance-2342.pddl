(define (problem prob)
 (:domain barman)
 (:objects 
      shaker69 - shaker
      left right - hand
      shot364 - shot
      ingredient370 ingredient162 - ingredient
      cocktail418 - cocktail
      dispenser276 dispenser460 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker69)
  (ontable shot364)
  (dispenses dispenser276 ingredient370)
  (dispenses dispenser460 ingredient162)
  (clean shaker69)
  (clean shot364)
  (empty shaker69)
  (empty shot364)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker69 l0)
  (shaker-level shaker69 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail418 ingredient370)
  (cocktail-part2 cocktail418 ingredient162)
)
 (:goal
  (and
      (contains shot364 cocktail418)
)))
