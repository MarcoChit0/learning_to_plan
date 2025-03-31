(define (problem prob)
 (:domain barman)
 (:objects 
      shaker229 - shaker
      left right - hand
      shot280 - shot
      ingredient339 ingredient315 ingredient291 ingredient399 - ingredient
      cocktail252 - cocktail
      dispenser341 dispenser338 dispenser177 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker229)
  (ontable shot280)
  (dispenses dispenser341 ingredient339)
  (dispenses dispenser338 ingredient315)
  (dispenses dispenser177 ingredient291)
  (dispenses dispenser397 ingredient399)
  (clean shaker229)
  (clean shot280)
  (empty shaker229)
  (empty shot280)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker229 l0)
  (shaker-level shaker229 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail252 ingredient315)
  (cocktail-part2 cocktail252 ingredient399)
)
 (:goal
  (and
      (contains shot280 cocktail252)
)))
