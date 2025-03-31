(define (problem prob)
 (:domain barman)
 (:objects 
      shaker166 - shaker
      left right - hand
      shot426 shot361 shot415 - shot
      ingredient202 ingredient468 - ingredient
      cocktail130 - cocktail
      dispenser189 dispenser388 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker166)
  (ontable shot426)
  (ontable shot361)
  (ontable shot415)
  (dispenses dispenser189 ingredient202)
  (dispenses dispenser388 ingredient468)
  (clean shaker166)
  (clean shot426)
  (clean shot361)
  (clean shot415)
  (empty shaker166)
  (empty shot426)
  (empty shot361)
  (empty shot415)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker166 l0)
  (shaker-level shaker166 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail130 ingredient202)
  (cocktail-part2 cocktail130 ingredient468)
)
 (:goal
  (and
      (contains shot426 cocktail130)
      (contains shot361 cocktail130)
)))
