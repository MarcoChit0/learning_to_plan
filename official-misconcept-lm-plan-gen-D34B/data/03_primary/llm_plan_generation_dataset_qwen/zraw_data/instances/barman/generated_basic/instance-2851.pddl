(define (problem prob)
 (:domain barman)
 (:objects 
      shaker74 - shaker
      left right - hand
      shot441 shot176 - shot
      ingredient322 ingredient93 ingredient107 ingredient64 - ingredient
      cocktail305 - cocktail
      dispenser314 dispenser51 dispenser497 dispenser396 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker74)
  (ontable shot441)
  (ontable shot176)
  (dispenses dispenser314 ingredient322)
  (dispenses dispenser51 ingredient93)
  (dispenses dispenser497 ingredient107)
  (dispenses dispenser396 ingredient64)
  (clean shaker74)
  (clean shot441)
  (clean shot176)
  (empty shaker74)
  (empty shot441)
  (empty shot176)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker74 l0)
  (shaker-level shaker74 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail305 ingredient107)
  (cocktail-part2 cocktail305 ingredient93)
)
 (:goal
  (and
      (contains shot441 cocktail305)
)))
