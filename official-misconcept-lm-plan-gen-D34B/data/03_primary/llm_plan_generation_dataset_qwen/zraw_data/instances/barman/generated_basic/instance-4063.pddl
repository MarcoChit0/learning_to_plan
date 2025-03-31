(define (problem prob)
 (:domain barman)
 (:objects 
      shaker244 - shaker
      left right - hand
      shot141 shot50 - shot
      ingredient252 ingredient322 - ingredient
      cocktail6 - cocktail
      dispenser214 dispenser250 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker244)
  (ontable shot141)
  (ontable shot50)
  (dispenses dispenser214 ingredient252)
  (dispenses dispenser250 ingredient322)
  (clean shaker244)
  (clean shot141)
  (clean shot50)
  (empty shaker244)
  (empty shot141)
  (empty shot50)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker244 l0)
  (shaker-level shaker244 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail6 ingredient252)
  (cocktail-part2 cocktail6 ingredient322)
)
 (:goal
  (and
      (contains shot141 cocktail6)
)))
