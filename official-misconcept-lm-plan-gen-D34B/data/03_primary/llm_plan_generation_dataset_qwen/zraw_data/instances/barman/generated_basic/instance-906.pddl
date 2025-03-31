(define (problem prob)
 (:domain barman)
 (:objects 
      shaker195 - shaker
      left right - hand
      shot209 shot84 - shot
      ingredient305 ingredient184 - ingredient
      cocktail1 - cocktail
      dispenser331 dispenser55 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker195)
  (ontable shot209)
  (ontable shot84)
  (dispenses dispenser331 ingredient305)
  (dispenses dispenser55 ingredient184)
  (clean shaker195)
  (clean shot209)
  (clean shot84)
  (empty shaker195)
  (empty shot209)
  (empty shot84)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker195 l0)
  (shaker-level shaker195 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient184)
  (cocktail-part2 cocktail1 ingredient305)
)
 (:goal
  (and
      (contains shot209 cocktail1)
)))
