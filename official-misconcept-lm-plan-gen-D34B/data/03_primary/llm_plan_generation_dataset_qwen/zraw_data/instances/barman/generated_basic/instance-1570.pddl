(define (problem prob)
 (:domain barman)
 (:objects 
      shaker306 - shaker
      left right - hand
      shot181 shot263 shot43 - shot
      ingredient66 ingredient406 ingredient318 ingredient329 - ingredient
      cocktail483 - cocktail
      dispenser327 dispenser331 dispenser159 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker306)
  (ontable shot181)
  (ontable shot263)
  (ontable shot43)
  (dispenses dispenser327 ingredient66)
  (dispenses dispenser331 ingredient406)
  (dispenses dispenser159 ingredient318)
  (dispenses dispenser306 ingredient329)
  (clean shaker306)
  (clean shot181)
  (clean shot263)
  (clean shot43)
  (empty shaker306)
  (empty shot181)
  (empty shot263)
  (empty shot43)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker306 l0)
  (shaker-level shaker306 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail483 ingredient406)
  (cocktail-part2 cocktail483 ingredient66)
)
 (:goal
  (and
      (contains shot181 cocktail483)
      (contains shot263 cocktail483)
)))
