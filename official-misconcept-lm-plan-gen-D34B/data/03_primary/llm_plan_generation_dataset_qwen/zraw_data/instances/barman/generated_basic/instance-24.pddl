(define (problem prob)
 (:domain barman)
 (:objects 
      shaker442 - shaker
      left right - hand
      shot101 shot63 - shot
      ingredient322 ingredient414 - ingredient
      cocktail1 - cocktail
      dispenser322 dispenser441 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker442)
  (ontable shot101)
  (ontable shot63)
  (dispenses dispenser322 ingredient322)
  (dispenses dispenser441 ingredient414)
  (clean shaker442)
  (clean shot101)
  (clean shot63)
  (empty shaker442)
  (empty shot101)
  (empty shot63)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker442 l0)
  (shaker-level shaker442 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient414)
  (cocktail-part2 cocktail1 ingredient322)
)
 (:goal
  (and
      (contains shot101 cocktail1)
)))
