(define (problem prob)
 (:domain barman)
 (:objects 
      shaker319 - shaker
      left right - hand
      shot184 shot382 shot4 - shot
      ingredient22 ingredient257 ingredient410 ingredient65 - ingredient
      cocktail1 - cocktail
      dispenser464 dispenser354 dispenser132 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker319)
  (ontable shot184)
  (ontable shot382)
  (ontable shot4)
  (dispenses dispenser464 ingredient22)
  (dispenses dispenser354 ingredient257)
  (dispenses dispenser132 ingredient410)
  (dispenses dispenser150 ingredient65)
  (clean shaker319)
  (clean shot184)
  (clean shot382)
  (clean shot4)
  (empty shaker319)
  (empty shot184)
  (empty shot382)
  (empty shot4)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker319 l0)
  (shaker-level shaker319 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient22)
  (cocktail-part2 cocktail1 ingredient65)
)
 (:goal
  (and
      (contains shot184 cocktail1)
      (contains shot382 cocktail1)
)))
