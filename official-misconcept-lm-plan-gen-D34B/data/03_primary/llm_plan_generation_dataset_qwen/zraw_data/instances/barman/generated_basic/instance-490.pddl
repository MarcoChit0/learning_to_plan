(define (problem prob)
 (:domain barman)
 (:objects 
      shaker432 - shaker
      left right - hand
      shot31 shot444 - shot
      ingredient415 ingredient247 ingredient110 ingredient46 - ingredient
      cocktail1 - cocktail
      dispenser306 dispenser299 dispenser231 dispenser141 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker432)
  (ontable shot31)
  (ontable shot444)
  (dispenses dispenser306 ingredient415)
  (dispenses dispenser299 ingredient247)
  (dispenses dispenser231 ingredient110)
  (dispenses dispenser141 ingredient46)
  (clean shaker432)
  (clean shot31)
  (clean shot444)
  (empty shaker432)
  (empty shot31)
  (empty shot444)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker432 l0)
  (shaker-level shaker432 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient247)
  (cocktail-part2 cocktail1 ingredient110)
)
 (:goal
  (and
      (contains shot31 cocktail1)
)))
