(define (problem prob)
 (:domain barman)
 (:objects 
      shaker207 - shaker
      left right - hand
      shot142 - shot
      ingredient452 ingredient382 ingredient306 - ingredient
      cocktail222 - cocktail
      dispenser269 dispenser18 dispenser125 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker207)
  (ontable shot142)
  (dispenses dispenser269 ingredient452)
  (dispenses dispenser18 ingredient382)
  (dispenses dispenser125 ingredient306)
  (clean shaker207)
  (clean shot142)
  (empty shaker207)
  (empty shot142)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker207 l0)
  (shaker-level shaker207 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail222 ingredient452)
  (cocktail-part2 cocktail222 ingredient306)
)
 (:goal
  (and
      (contains shot142 cocktail222)
)))
