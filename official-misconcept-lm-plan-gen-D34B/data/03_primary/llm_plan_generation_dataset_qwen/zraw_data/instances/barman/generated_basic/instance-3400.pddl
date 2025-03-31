(define (problem prob)
 (:domain barman)
 (:objects 
      shaker466 - shaker
      left right - hand
      shot96 shot291 - shot
      ingredient283 ingredient94 ingredient79 ingredient383 - ingredient
      cocktail347 - cocktail
      dispenser385 dispenser45 dispenser244 dispenser387 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker466)
  (ontable shot96)
  (ontable shot291)
  (dispenses dispenser385 ingredient283)
  (dispenses dispenser45 ingredient94)
  (dispenses dispenser244 ingredient79)
  (dispenses dispenser387 ingredient383)
  (clean shaker466)
  (clean shot96)
  (clean shot291)
  (empty shaker466)
  (empty shot96)
  (empty shot291)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker466 l0)
  (shaker-level shaker466 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail347 ingredient383)
  (cocktail-part2 cocktail347 ingredient94)
)
 (:goal
  (and
      (contains shot96 cocktail347)
)))
