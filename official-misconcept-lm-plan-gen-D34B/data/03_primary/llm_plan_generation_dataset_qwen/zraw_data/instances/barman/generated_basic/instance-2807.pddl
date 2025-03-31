(define (problem prob)
 (:domain barman)
 (:objects 
      shaker412 - shaker
      left right - hand
      shot379 shot61 - shot
      ingredient389 ingredient224 - ingredient
      cocktail267 - cocktail
      dispenser288 dispenser255 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker412)
  (ontable shot379)
  (ontable shot61)
  (dispenses dispenser288 ingredient389)
  (dispenses dispenser255 ingredient224)
  (clean shaker412)
  (clean shot379)
  (clean shot61)
  (empty shaker412)
  (empty shot379)
  (empty shot61)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker412 l0)
  (shaker-level shaker412 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail267 ingredient389)
  (cocktail-part2 cocktail267 ingredient224)
)
 (:goal
  (and
      (contains shot379 cocktail267)
)))
