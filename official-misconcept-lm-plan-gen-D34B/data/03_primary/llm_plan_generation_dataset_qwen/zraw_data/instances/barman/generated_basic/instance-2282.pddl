(define (problem prob)
 (:domain barman)
 (:objects 
      shaker457 - shaker
      left right - hand
      shot483 - shot
      ingredient239 ingredient178 ingredient413 - ingredient
      cocktail330 - cocktail
      dispenser110 dispenser248 dispenser323 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker457)
  (ontable shot483)
  (dispenses dispenser110 ingredient239)
  (dispenses dispenser248 ingredient178)
  (dispenses dispenser323 ingredient413)
  (clean shaker457)
  (clean shot483)
  (empty shaker457)
  (empty shot483)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker457 l0)
  (shaker-level shaker457 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail330 ingredient239)
  (cocktail-part2 cocktail330 ingredient178)
)
 (:goal
  (and
      (contains shot483 cocktail330)
)))
