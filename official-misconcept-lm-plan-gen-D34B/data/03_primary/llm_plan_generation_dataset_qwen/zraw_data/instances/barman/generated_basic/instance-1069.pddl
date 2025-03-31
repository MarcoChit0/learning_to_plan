(define (problem prob)
 (:domain barman)
 (:objects 
      shaker202 - shaker
      left right - hand
      shot419 shot12 shot396 - shot
      ingredient255 ingredient151 ingredient315 ingredient147 - ingredient
      cocktail1 - cocktail
      dispenser65 dispenser184 dispenser470 dispenser336 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker202)
  (ontable shot419)
  (ontable shot12)
  (ontable shot396)
  (dispenses dispenser65 ingredient255)
  (dispenses dispenser184 ingredient151)
  (dispenses dispenser470 ingredient315)
  (dispenses dispenser336 ingredient147)
  (clean shaker202)
  (clean shot419)
  (clean shot12)
  (clean shot396)
  (empty shaker202)
  (empty shot419)
  (empty shot12)
  (empty shot396)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker202 l0)
  (shaker-level shaker202 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient151)
  (cocktail-part2 cocktail1 ingredient255)
)
 (:goal
  (and
      (contains shot419 cocktail1)
      (contains shot12 cocktail1)
)))
