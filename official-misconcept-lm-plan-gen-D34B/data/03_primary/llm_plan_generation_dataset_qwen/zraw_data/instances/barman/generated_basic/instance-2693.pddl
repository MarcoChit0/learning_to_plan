(define (problem prob)
 (:domain barman)
 (:objects 
      shaker56 - shaker
      left right - hand
      shot101 shot229 - shot
      ingredient309 ingredient463 ingredient208 - ingredient
      cocktail172 - cocktail
      dispenser47 dispenser144 dispenser103 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker56)
  (ontable shot101)
  (ontable shot229)
  (dispenses dispenser47 ingredient309)
  (dispenses dispenser144 ingredient463)
  (dispenses dispenser103 ingredient208)
  (clean shaker56)
  (clean shot101)
  (clean shot229)
  (empty shaker56)
  (empty shot101)
  (empty shot229)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker56 l0)
  (shaker-level shaker56 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail172 ingredient309)
  (cocktail-part2 cocktail172 ingredient463)
)
 (:goal
  (and
      (contains shot101 cocktail172)
)))
