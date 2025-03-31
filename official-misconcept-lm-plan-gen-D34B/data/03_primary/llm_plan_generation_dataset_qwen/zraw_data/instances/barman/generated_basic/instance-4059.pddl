(define (problem prob)
 (:domain barman)
 (:objects 
      shaker28 - shaker
      left right - hand
      shot441 shot314 - shot
      ingredient49 ingredient42 - ingredient
      cocktail194 - cocktail
      dispenser233 dispenser187 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker28)
  (ontable shot441)
  (ontable shot314)
  (dispenses dispenser233 ingredient49)
  (dispenses dispenser187 ingredient42)
  (clean shaker28)
  (clean shot441)
  (clean shot314)
  (empty shaker28)
  (empty shot441)
  (empty shot314)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker28 l0)
  (shaker-level shaker28 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail194 ingredient49)
  (cocktail-part2 cocktail194 ingredient42)
)
 (:goal
  (and
      (contains shot441 cocktail194)
)))
