(define (problem prob)
 (:domain barman)
 (:objects 
      shaker472 - shaker
      left right - hand
      shot307 shot172 - shot
      ingredient195 ingredient365 ingredient424 ingredient157 - ingredient
      cocktail80 - cocktail
      dispenser256 dispenser140 dispenser36 dispenser201 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker472)
  (ontable shot307)
  (ontable shot172)
  (dispenses dispenser256 ingredient195)
  (dispenses dispenser140 ingredient365)
  (dispenses dispenser36 ingredient424)
  (dispenses dispenser201 ingredient157)
  (clean shaker472)
  (clean shot307)
  (clean shot172)
  (empty shaker472)
  (empty shot307)
  (empty shot172)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker472 l0)
  (shaker-level shaker472 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail80 ingredient195)
  (cocktail-part2 cocktail80 ingredient365)
)
 (:goal
  (and
      (contains shot307 cocktail80)
)))
