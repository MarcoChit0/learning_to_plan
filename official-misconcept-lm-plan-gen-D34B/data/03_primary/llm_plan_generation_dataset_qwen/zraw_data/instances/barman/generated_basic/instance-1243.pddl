(define (problem prob)
 (:domain barman)
 (:objects 
      shaker101 - shaker
      left right - hand
      shot180 shot460 shot428 - shot
      ingredient476 ingredient221 ingredient436 ingredient347 - ingredient
      cocktail421 - cocktail
      dispenser450 dispenser24 dispenser298 dispenser325 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker101)
  (ontable shot180)
  (ontable shot460)
  (ontable shot428)
  (dispenses dispenser450 ingredient476)
  (dispenses dispenser24 ingredient221)
  (dispenses dispenser298 ingredient436)
  (dispenses dispenser325 ingredient347)
  (clean shaker101)
  (clean shot180)
  (clean shot460)
  (clean shot428)
  (empty shaker101)
  (empty shot180)
  (empty shot460)
  (empty shot428)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker101 l0)
  (shaker-level shaker101 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail421 ingredient436)
  (cocktail-part2 cocktail421 ingredient221)
)
 (:goal
  (and
      (contains shot180 cocktail421)
      (contains shot460 cocktail421)
)))
