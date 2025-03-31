(define (problem prob)
 (:domain barman)
 (:objects 
      shaker63 - shaker
      left right - hand
      shot420 shot127 - shot
      ingredient235 ingredient36 ingredient153 - ingredient
      cocktail1 - cocktail
      dispenser490 dispenser376 dispenser82 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker63)
  (ontable shot420)
  (ontable shot127)
  (dispenses dispenser490 ingredient235)
  (dispenses dispenser376 ingredient36)
  (dispenses dispenser82 ingredient153)
  (clean shaker63)
  (clean shot420)
  (clean shot127)
  (empty shaker63)
  (empty shot420)
  (empty shot127)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker63 l0)
  (shaker-level shaker63 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient36)
  (cocktail-part2 cocktail1 ingredient153)
)
 (:goal
  (and
      (contains shot420 cocktail1)
)))
