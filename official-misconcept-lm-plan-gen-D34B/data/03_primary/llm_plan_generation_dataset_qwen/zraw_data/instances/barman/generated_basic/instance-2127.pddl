(define (problem prob)
 (:domain barman)
 (:objects 
      shaker177 - shaker
      left right - hand
      shot490 shot184 shot32 - shot
      ingredient168 ingredient182 - ingredient
      cocktail493 - cocktail
      dispenser200 dispenser0 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker177)
  (ontable shot490)
  (ontable shot184)
  (ontable shot32)
  (dispenses dispenser200 ingredient168)
  (dispenses dispenser0 ingredient182)
  (clean shaker177)
  (clean shot490)
  (clean shot184)
  (clean shot32)
  (empty shaker177)
  (empty shot490)
  (empty shot184)
  (empty shot32)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker177 l0)
  (shaker-level shaker177 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail493 ingredient182)
  (cocktail-part2 cocktail493 ingredient168)
)
 (:goal
  (and
      (contains shot490 cocktail493)
      (contains shot184 ingredient182)
)))
