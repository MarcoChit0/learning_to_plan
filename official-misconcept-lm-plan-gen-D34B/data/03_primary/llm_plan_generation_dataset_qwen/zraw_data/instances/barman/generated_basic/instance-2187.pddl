(define (problem prob)
 (:domain barman)
 (:objects 
      shaker318 - shaker
      left right - hand
      shot321 shot282 shot304 - shot
      ingredient280 ingredient485 - ingredient
      cocktail101 - cocktail
      dispenser187 dispenser460 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker318)
  (ontable shot321)
  (ontable shot282)
  (ontable shot304)
  (dispenses dispenser187 ingredient280)
  (dispenses dispenser460 ingredient485)
  (clean shaker318)
  (clean shot321)
  (clean shot282)
  (clean shot304)
  (empty shaker318)
  (empty shot321)
  (empty shot282)
  (empty shot304)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker318 l0)
  (shaker-level shaker318 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail101 ingredient280)
  (cocktail-part2 cocktail101 ingredient485)
)
 (:goal
  (and
      (contains shot321 cocktail101)
      (contains shot282 ingredient280)
)))
