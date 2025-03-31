(define (problem prob)
 (:domain barman)
 (:objects 
      shaker490 - shaker
      left right - hand
      shot202 shot153 - shot
      ingredient155 ingredient324 ingredient129 - ingredient
      cocktail1 - cocktail
      dispenser302 dispenser62 dispenser315 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker490)
  (ontable shot202)
  (ontable shot153)
  (dispenses dispenser302 ingredient155)
  (dispenses dispenser62 ingredient324)
  (dispenses dispenser315 ingredient129)
  (clean shaker490)
  (clean shot202)
  (clean shot153)
  (empty shaker490)
  (empty shot202)
  (empty shot153)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker490 l0)
  (shaker-level shaker490 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient324)
  (cocktail-part2 cocktail1 ingredient155)
)
 (:goal
  (and
      (contains shot202 cocktail1)
)))
