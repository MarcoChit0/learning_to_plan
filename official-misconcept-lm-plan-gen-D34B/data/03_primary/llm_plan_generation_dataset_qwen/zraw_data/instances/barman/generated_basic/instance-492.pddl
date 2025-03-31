(define (problem prob)
 (:domain barman)
 (:objects 
      shaker123 - shaker
      left right - hand
      shot343 shot134 - shot
      ingredient203 ingredient171 ingredient326 ingredient259 - ingredient
      cocktail1 - cocktail
      dispenser349 dispenser472 dispenser172 dispenser438 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker123)
  (ontable shot343)
  (ontable shot134)
  (dispenses dispenser349 ingredient203)
  (dispenses dispenser472 ingredient171)
  (dispenses dispenser172 ingredient326)
  (dispenses dispenser438 ingredient259)
  (clean shaker123)
  (clean shot343)
  (clean shot134)
  (empty shaker123)
  (empty shot343)
  (empty shot134)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker123 l0)
  (shaker-level shaker123 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient326)
  (cocktail-part2 cocktail1 ingredient259)
)
 (:goal
  (and
      (contains shot343 cocktail1)
)))
