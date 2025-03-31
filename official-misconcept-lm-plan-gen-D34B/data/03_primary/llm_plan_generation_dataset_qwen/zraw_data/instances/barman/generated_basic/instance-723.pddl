(define (problem prob)
 (:domain barman)
 (:objects 
      shaker240 - shaker
      left right - hand
      shot240 - shot
      ingredient25 ingredient258 ingredient137 - ingredient
      cocktail1 - cocktail
      dispenser371 dispenser56 dispenser268 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker240)
  (ontable shot240)
  (dispenses dispenser371 ingredient25)
  (dispenses dispenser56 ingredient258)
  (dispenses dispenser268 ingredient137)
  (clean shaker240)
  (clean shot240)
  (empty shaker240)
  (empty shot240)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker240 l0)
  (shaker-level shaker240 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient137)
  (cocktail-part2 cocktail1 ingredient258)
)
 (:goal
  (and
      (contains shot240 cocktail1)
)))
