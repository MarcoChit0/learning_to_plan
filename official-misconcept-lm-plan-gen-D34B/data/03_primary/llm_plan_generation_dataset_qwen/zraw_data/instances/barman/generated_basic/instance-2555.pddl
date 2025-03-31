(define (problem prob)
 (:domain barman)
 (:objects 
      shaker442 - shaker
      left right - hand
      shot382 shot33 shot12 - shot
      ingredient433 ingredient438 - ingredient
      cocktail275 - cocktail
      dispenser236 dispenser151 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker442)
  (ontable shot382)
  (ontable shot33)
  (ontable shot12)
  (dispenses dispenser236 ingredient433)
  (dispenses dispenser151 ingredient438)
  (clean shaker442)
  (clean shot382)
  (clean shot33)
  (clean shot12)
  (empty shaker442)
  (empty shot382)
  (empty shot33)
  (empty shot12)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker442 l0)
  (shaker-level shaker442 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail275 ingredient433)
  (cocktail-part2 cocktail275 ingredient438)
)
 (:goal
  (and
      (contains shot382 cocktail275)
      (contains shot33 cocktail275)
)))
