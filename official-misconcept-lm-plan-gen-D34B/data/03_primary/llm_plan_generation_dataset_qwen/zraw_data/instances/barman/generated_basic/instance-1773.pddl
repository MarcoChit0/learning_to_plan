(define (problem prob)
 (:domain barman)
 (:objects 
      shaker187 - shaker
      left right - hand
      shot347 shot17 shot270 - shot
      ingredient249 ingredient433 - ingredient
      cocktail479 - cocktail
      dispenser448 dispenser482 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker187)
  (ontable shot347)
  (ontable shot17)
  (ontable shot270)
  (dispenses dispenser448 ingredient249)
  (dispenses dispenser482 ingredient433)
  (clean shaker187)
  (clean shot347)
  (clean shot17)
  (clean shot270)
  (empty shaker187)
  (empty shot347)
  (empty shot17)
  (empty shot270)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker187 l0)
  (shaker-level shaker187 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail479 ingredient433)
  (cocktail-part2 cocktail479 ingredient249)
)
 (:goal
  (and
      (contains shot347 cocktail479)
      (contains shot17 ingredient249)
)))
