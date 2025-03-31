(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot205 - shot
      ingredient387 ingredient382 ingredient355 ingredient261 - ingredient
      cocktail405 - cocktail
      dispenser413 dispenser143 dispenser434 dispenser186 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot205)
  (dispenses dispenser413 ingredient387)
  (dispenses dispenser143 ingredient382)
  (dispenses dispenser434 ingredient355)
  (dispenses dispenser186 ingredient261)
  (clean shaker221)
  (clean shot205)
  (empty shaker221)
  (empty shot205)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail405 ingredient387)
  (cocktail-part2 cocktail405 ingredient355)
)
 (:goal
  (and
      (contains shot205 cocktail405)
)))
