(define (problem prob)
 (:domain barman)
 (:objects 
      shaker275 - shaker
      left right - hand
      shot233 shot162 - shot
      ingredient405 ingredient186 ingredient8 ingredient318 - ingredient
      cocktail304 - cocktail
      dispenser19 dispenser47 dispenser125 dispenser440 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker275)
  (ontable shot233)
  (ontable shot162)
  (dispenses dispenser19 ingredient405)
  (dispenses dispenser47 ingredient186)
  (dispenses dispenser125 ingredient8)
  (dispenses dispenser440 ingredient318)
  (clean shaker275)
  (clean shot233)
  (clean shot162)
  (empty shaker275)
  (empty shot233)
  (empty shot162)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker275 l0)
  (shaker-level shaker275 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail304 ingredient318)
  (cocktail-part2 cocktail304 ingredient8)
)
 (:goal
  (and
      (contains shot233 cocktail304)
)))
