(define (problem prob)
 (:domain barman)
 (:objects 
      shaker418 - shaker
      left right - hand
      shot220 - shot
      ingredient438 ingredient206 ingredient188 ingredient30 - ingredient
      cocktail127 - cocktail
      dispenser325 dispenser447 dispenser358 dispenser145 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker418)
  (ontable shot220)
  (dispenses dispenser325 ingredient438)
  (dispenses dispenser447 ingredient206)
  (dispenses dispenser358 ingredient188)
  (dispenses dispenser145 ingredient30)
  (clean shaker418)
  (clean shot220)
  (empty shaker418)
  (empty shot220)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker418 l0)
  (shaker-level shaker418 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail127 ingredient188)
  (cocktail-part2 cocktail127 ingredient438)
)
 (:goal
  (and
      (contains shot220 cocktail127)
)))
