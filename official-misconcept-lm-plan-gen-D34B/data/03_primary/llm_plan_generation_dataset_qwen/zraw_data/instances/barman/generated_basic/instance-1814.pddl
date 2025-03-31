(define (problem prob)
 (:domain barman)
 (:objects 
      shaker246 - shaker
      left right - hand
      shot318 shot474 shot182 - shot
      ingredient202 ingredient119 ingredient90 ingredient386 - ingredient
      cocktail9 - cocktail
      dispenser104 dispenser25 dispenser270 dispenser98 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker246)
  (ontable shot318)
  (ontable shot474)
  (ontable shot182)
  (dispenses dispenser104 ingredient202)
  (dispenses dispenser25 ingredient119)
  (dispenses dispenser270 ingredient90)
  (dispenses dispenser98 ingredient386)
  (clean shaker246)
  (clean shot318)
  (clean shot474)
  (clean shot182)
  (empty shaker246)
  (empty shot318)
  (empty shot474)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker246 l0)
  (shaker-level shaker246 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail9 ingredient90)
  (cocktail-part2 cocktail9 ingredient202)
)
 (:goal
  (and
      (contains shot318 cocktail9)
      (contains shot474 cocktail9)
)))
