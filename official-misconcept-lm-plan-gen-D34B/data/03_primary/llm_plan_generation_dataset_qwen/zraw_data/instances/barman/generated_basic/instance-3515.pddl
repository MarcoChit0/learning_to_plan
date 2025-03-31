(define (problem prob)
 (:domain barman)
 (:objects 
      shaker21 - shaker
      left right - hand
      shot4 shot74 - shot
      ingredient250 ingredient337 ingredient94 ingredient87 - ingredient
      cocktail365 - cocktail
      dispenser79 dispenser363 dispenser350 dispenser15 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker21)
  (ontable shot4)
  (ontable shot74)
  (dispenses dispenser79 ingredient250)
  (dispenses dispenser363 ingredient337)
  (dispenses dispenser350 ingredient94)
  (dispenses dispenser15 ingredient87)
  (clean shaker21)
  (clean shot4)
  (clean shot74)
  (empty shaker21)
  (empty shot4)
  (empty shot74)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker21 l0)
  (shaker-level shaker21 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail365 ingredient337)
  (cocktail-part2 cocktail365 ingredient250)
)
 (:goal
  (and
      (contains shot4 cocktail365)
)))
