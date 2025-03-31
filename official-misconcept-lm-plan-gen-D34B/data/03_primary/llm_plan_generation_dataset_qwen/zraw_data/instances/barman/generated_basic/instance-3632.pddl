(define (problem prob)
 (:domain barman)
 (:objects 
      shaker344 - shaker
      left right - hand
      shot236 - shot
      ingredient215 ingredient32 ingredient6 ingredient12 - ingredient
      cocktail113 - cocktail
      dispenser417 dispenser421 dispenser308 dispenser138 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker344)
  (ontable shot236)
  (dispenses dispenser417 ingredient215)
  (dispenses dispenser421 ingredient32)
  (dispenses dispenser308 ingredient6)
  (dispenses dispenser138 ingredient12)
  (clean shaker344)
  (clean shot236)
  (empty shaker344)
  (empty shot236)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker344 l0)
  (shaker-level shaker344 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail113 ingredient12)
  (cocktail-part2 cocktail113 ingredient6)
)
 (:goal
  (and
      (contains shot236 cocktail113)
)))
