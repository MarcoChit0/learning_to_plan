(define (problem prob)
 (:domain barman)
 (:objects 
      shaker439 - shaker
      left right - hand
      shot175 shot431 - shot
      ingredient495 ingredient137 ingredient255 ingredient40 - ingredient
      cocktail331 - cocktail
      dispenser311 dispenser248 dispenser126 dispenser20 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker439)
  (ontable shot175)
  (ontable shot431)
  (dispenses dispenser311 ingredient495)
  (dispenses dispenser248 ingredient137)
  (dispenses dispenser126 ingredient255)
  (dispenses dispenser20 ingredient40)
  (clean shaker439)
  (clean shot175)
  (clean shot431)
  (empty shaker439)
  (empty shot175)
  (empty shot431)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker439 l0)
  (shaker-level shaker439 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail331 ingredient255)
  (cocktail-part2 cocktail331 ingredient495)
)
 (:goal
  (and
      (contains shot175 cocktail331)
)))
