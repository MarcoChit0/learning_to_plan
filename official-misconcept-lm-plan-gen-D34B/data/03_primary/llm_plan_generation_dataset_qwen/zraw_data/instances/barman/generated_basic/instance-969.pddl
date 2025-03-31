(define (problem prob)
 (:domain barman)
 (:objects 
      shaker449 - shaker
      left right - hand
      shot135 shot483 - shot
      ingredient99 ingredient473 ingredient51 ingredient381 - ingredient
      cocktail1 - cocktail
      dispenser386 dispenser248 dispenser123 dispenser110 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker449)
  (ontable shot135)
  (ontable shot483)
  (dispenses dispenser386 ingredient99)
  (dispenses dispenser248 ingredient473)
  (dispenses dispenser123 ingredient51)
  (dispenses dispenser110 ingredient381)
  (clean shaker449)
  (clean shot135)
  (clean shot483)
  (empty shaker449)
  (empty shot135)
  (empty shot483)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker449 l0)
  (shaker-level shaker449 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient99)
  (cocktail-part2 cocktail1 ingredient473)
)
 (:goal
  (and
      (contains shot135 cocktail1)
)))
