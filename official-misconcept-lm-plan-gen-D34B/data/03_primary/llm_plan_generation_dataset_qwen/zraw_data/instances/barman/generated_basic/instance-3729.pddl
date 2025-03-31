(define (problem prob)
 (:domain barman)
 (:objects 
      shaker433 - shaker
      left right - hand
      shot426 - shot
      ingredient121 ingredient314 ingredient315 ingredient41 - ingredient
      cocktail269 - cocktail
      dispenser6 dispenser404 dispenser191 dispenser32 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker433)
  (ontable shot426)
  (dispenses dispenser6 ingredient121)
  (dispenses dispenser404 ingredient314)
  (dispenses dispenser191 ingredient315)
  (dispenses dispenser32 ingredient41)
  (clean shaker433)
  (clean shot426)
  (empty shaker433)
  (empty shot426)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker433 l0)
  (shaker-level shaker433 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail269 ingredient121)
  (cocktail-part2 cocktail269 ingredient41)
)
 (:goal
  (and
      (contains shot426 cocktail269)
)))
