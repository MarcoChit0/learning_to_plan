(define (problem prob)
 (:domain barman)
 (:objects 
      shaker116 - shaker
      left right - hand
      shot243 shot458 - shot
      ingredient293 ingredient48 ingredient360 ingredient85 - ingredient
      cocktail27 - cocktail
      dispenser477 dispenser431 dispenser179 dispenser82 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker116)
  (ontable shot243)
  (ontable shot458)
  (dispenses dispenser477 ingredient293)
  (dispenses dispenser431 ingredient48)
  (dispenses dispenser179 ingredient360)
  (dispenses dispenser82 ingredient85)
  (clean shaker116)
  (clean shot243)
  (clean shot458)
  (empty shaker116)
  (empty shot243)
  (empty shot458)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker116 l0)
  (shaker-level shaker116 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail27 ingredient293)
  (cocktail-part2 cocktail27 ingredient85)
)
 (:goal
  (and
      (contains shot243 cocktail27)
)))
