(define (problem prob)
 (:domain barman)
 (:objects 
      shaker6 - shaker
      left right - hand
      shot21 shot120 - shot
      ingredient85 ingredient117 ingredient321 ingredient224 - ingredient
      cocktail89 - cocktail
      dispenser117 dispenser417 dispenser320 dispenser80 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker6)
  (ontable shot21)
  (ontable shot120)
  (dispenses dispenser117 ingredient85)
  (dispenses dispenser417 ingredient117)
  (dispenses dispenser320 ingredient321)
  (dispenses dispenser80 ingredient224)
  (clean shaker6)
  (clean shot21)
  (clean shot120)
  (empty shaker6)
  (empty shot21)
  (empty shot120)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker6 l0)
  (shaker-level shaker6 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail89 ingredient224)
  (cocktail-part2 cocktail89 ingredient117)
)
 (:goal
  (and
      (contains shot21 cocktail89)
)))
