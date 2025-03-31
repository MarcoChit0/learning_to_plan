(define (problem prob)
 (:domain barman)
 (:objects 
      shaker399 - shaker
      left right - hand
      shot212 shot385 - shot
      ingredient317 ingredient495 ingredient59 ingredient178 - ingredient
      cocktail178 - cocktail
      dispenser284 dispenser202 dispenser435 dispenser498 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker399)
  (ontable shot212)
  (ontable shot385)
  (dispenses dispenser284 ingredient317)
  (dispenses dispenser202 ingredient495)
  (dispenses dispenser435 ingredient59)
  (dispenses dispenser498 ingredient178)
  (clean shaker399)
  (clean shot212)
  (clean shot385)
  (empty shaker399)
  (empty shot212)
  (empty shot385)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker399 l0)
  (shaker-level shaker399 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail178 ingredient317)
  (cocktail-part2 cocktail178 ingredient59)
)
 (:goal
  (and
      (contains shot212 cocktail178)
)))
