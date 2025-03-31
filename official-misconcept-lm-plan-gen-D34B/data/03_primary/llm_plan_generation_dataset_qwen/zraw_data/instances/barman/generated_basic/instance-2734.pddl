(define (problem prob)
 (:domain barman)
 (:objects 
      shaker349 - shaker
      left right - hand
      shot193 shot315 shot327 - shot
      ingredient398 ingredient236 ingredient25 - ingredient
      cocktail40 - cocktail
      dispenser477 dispenser217 dispenser139 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker349)
  (ontable shot193)
  (ontable shot315)
  (ontable shot327)
  (dispenses dispenser477 ingredient398)
  (dispenses dispenser217 ingredient236)
  (dispenses dispenser139 ingredient25)
  (clean shaker349)
  (clean shot193)
  (clean shot315)
  (clean shot327)
  (empty shaker349)
  (empty shot193)
  (empty shot315)
  (empty shot327)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker349 l0)
  (shaker-level shaker349 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail40 ingredient398)
  (cocktail-part2 cocktail40 ingredient236)
)
 (:goal
  (and
      (contains shot193 cocktail40)
      (contains shot315 cocktail40)
)))
