(define (problem prob)
 (:domain barman)
 (:objects 
      shaker381 - shaker
      left right - hand
      shot343 - shot
      ingredient323 ingredient288 ingredient358 - ingredient
      cocktail125 - cocktail
      dispenser254 dispenser182 dispenser321 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker381)
  (ontable shot343)
  (dispenses dispenser254 ingredient323)
  (dispenses dispenser182 ingredient288)
  (dispenses dispenser321 ingredient358)
  (clean shaker381)
  (clean shot343)
  (empty shaker381)
  (empty shot343)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker381 l0)
  (shaker-level shaker381 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail125 ingredient288)
  (cocktail-part2 cocktail125 ingredient323)
)
 (:goal
  (and
      (contains shot343 cocktail125)
)))
