(define (problem prob)
 (:domain barman)
 (:objects 
      shaker21 - shaker
      left right - hand
      shot105 shot74 - shot
      ingredient45 ingredient406 ingredient322 ingredient123 - ingredient
      cocktail157 - cocktail
      dispenser427 dispenser275 dispenser463 dispenser489 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker21)
  (ontable shot105)
  (ontable shot74)
  (dispenses dispenser427 ingredient45)
  (dispenses dispenser275 ingredient406)
  (dispenses dispenser463 ingredient322)
  (dispenses dispenser489 ingredient123)
  (clean shaker21)
  (clean shot105)
  (clean shot74)
  (empty shaker21)
  (empty shot105)
  (empty shot74)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker21 l0)
  (shaker-level shaker21 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail157 ingredient406)
  (cocktail-part2 cocktail157 ingredient322)
)
 (:goal
  (and
      (contains shot105 cocktail157)
)))
