(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot331 shot394 - shot
      ingredient10 ingredient215 ingredient195 ingredient490 - ingredient
      cocktail422 - cocktail
      dispenser300 dispenser6 dispenser438 dispenser120 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot331)
  (ontable shot394)
  (dispenses dispenser300 ingredient10)
  (dispenses dispenser6 ingredient215)
  (dispenses dispenser438 ingredient195)
  (dispenses dispenser120 ingredient490)
  (clean shaker236)
  (clean shot331)
  (clean shot394)
  (empty shaker236)
  (empty shot331)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail422 ingredient490)
  (cocktail-part2 cocktail422 ingredient10)
)
 (:goal
  (and
      (contains shot331 cocktail422)
)))
