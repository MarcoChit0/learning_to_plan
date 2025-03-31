(define (problem prob)
 (:domain barman)
 (:objects 
      shaker246 - shaker
      left right - hand
      shot475 - shot
      ingredient466 ingredient230 ingredient478 ingredient153 - ingredient
      cocktail388 - cocktail
      dispenser245 dispenser50 dispenser132 dispenser22 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker246)
  (ontable shot475)
  (dispenses dispenser245 ingredient466)
  (dispenses dispenser50 ingredient230)
  (dispenses dispenser132 ingredient478)
  (dispenses dispenser22 ingredient153)
  (clean shaker246)
  (clean shot475)
  (empty shaker246)
  (empty shot475)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker246 l0)
  (shaker-level shaker246 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail388 ingredient478)
  (cocktail-part2 cocktail388 ingredient230)
)
 (:goal
  (and
      (contains shot475 cocktail388)
)))
