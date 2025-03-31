(define (problem prob)
 (:domain barman)
 (:objects 
      shaker365 - shaker
      left right - hand
      shot462 - shot
      ingredient385 ingredient130 ingredient284 ingredient49 - ingredient
      cocktail1 - cocktail
      dispenser272 dispenser21 dispenser463 dispenser470 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker365)
  (ontable shot462)
  (dispenses dispenser272 ingredient385)
  (dispenses dispenser21 ingredient130)
  (dispenses dispenser463 ingredient284)
  (dispenses dispenser470 ingredient49)
  (clean shaker365)
  (clean shot462)
  (empty shaker365)
  (empty shot462)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker365 l0)
  (shaker-level shaker365 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient130)
  (cocktail-part2 cocktail1 ingredient49)
)
 (:goal
  (and
      (contains shot462 cocktail1)
)))
