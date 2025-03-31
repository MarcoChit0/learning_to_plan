(define (problem prob)
 (:domain barman)
 (:objects 
      shaker103 - shaker
      left right - hand
      shot135 shot78 - shot
      ingredient225 ingredient423 ingredient26 - ingredient
      cocktail412 - cocktail
      dispenser190 dispenser132 dispenser465 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker103)
  (ontable shot135)
  (ontable shot78)
  (dispenses dispenser190 ingredient225)
  (dispenses dispenser132 ingredient423)
  (dispenses dispenser465 ingredient26)
  (clean shaker103)
  (clean shot135)
  (clean shot78)
  (empty shaker103)
  (empty shot135)
  (empty shot78)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker103 l0)
  (shaker-level shaker103 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail412 ingredient26)
  (cocktail-part2 cocktail412 ingredient423)
)
 (:goal
  (and
      (contains shot135 cocktail412)
)))
