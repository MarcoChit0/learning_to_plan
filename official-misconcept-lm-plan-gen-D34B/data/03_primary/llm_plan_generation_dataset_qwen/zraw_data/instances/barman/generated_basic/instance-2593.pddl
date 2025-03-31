(define (problem prob)
 (:domain barman)
 (:objects 
      shaker55 - shaker
      left right - hand
      shot120 shot189 - shot
      ingredient286 ingredient381 ingredient470 - ingredient
      cocktail385 - cocktail
      dispenser46 dispenser302 dispenser104 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker55)
  (ontable shot120)
  (ontable shot189)
  (dispenses dispenser46 ingredient286)
  (dispenses dispenser302 ingredient381)
  (dispenses dispenser104 ingredient470)
  (clean shaker55)
  (clean shot120)
  (clean shot189)
  (empty shaker55)
  (empty shot120)
  (empty shot189)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker55 l0)
  (shaker-level shaker55 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail385 ingredient470)
  (cocktail-part2 cocktail385 ingredient286)
)
 (:goal
  (and
      (contains shot120 cocktail385)
)))
