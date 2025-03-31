(define (problem prob)
 (:domain barman)
 (:objects 
      shaker61 - shaker
      left right - hand
      shot291 - shot
      ingredient467 ingredient71 ingredient322 - ingredient
      cocktail221 - cocktail
      dispenser81 dispenser8 dispenser325 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker61)
  (ontable shot291)
  (dispenses dispenser81 ingredient467)
  (dispenses dispenser8 ingredient71)
  (dispenses dispenser325 ingredient322)
  (clean shaker61)
  (clean shot291)
  (empty shaker61)
  (empty shot291)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker61 l0)
  (shaker-level shaker61 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail221 ingredient322)
  (cocktail-part2 cocktail221 ingredient467)
)
 (:goal
  (and
      (contains shot291 cocktail221)
)))
