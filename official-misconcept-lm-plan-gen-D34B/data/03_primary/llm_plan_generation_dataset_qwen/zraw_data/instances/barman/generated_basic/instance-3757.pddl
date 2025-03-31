(define (problem prob)
 (:domain barman)
 (:objects 
      shaker208 - shaker
      left right - hand
      shot17 shot270 shot484 - shot
      ingredient173 ingredient411 - ingredient
      cocktail331 - cocktail
      dispenser228 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker208)
  (ontable shot17)
  (ontable shot270)
  (ontable shot484)
  (dispenses dispenser228 ingredient173)
  (dispenses dispenser290 ingredient411)
  (clean shaker208)
  (clean shot17)
  (clean shot270)
  (clean shot484)
  (empty shaker208)
  (empty shot17)
  (empty shot270)
  (empty shot484)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker208 l0)
  (shaker-level shaker208 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail331 ingredient411)
  (cocktail-part2 cocktail331 ingredient173)
)
 (:goal
  (and
      (contains shot17 cocktail331)
      (contains shot270 ingredient411)
)))
