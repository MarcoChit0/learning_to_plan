(define (problem prob)
 (:domain barman)
 (:objects 
      shaker181 - shaker
      left right - hand
      shot137 - shot
      ingredient122 ingredient128 ingredient293 ingredient394 - ingredient
      cocktail1 - cocktail
      dispenser353 dispenser385 dispenser137 dispenser354 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker181)
  (ontable shot137)
  (dispenses dispenser353 ingredient122)
  (dispenses dispenser385 ingredient128)
  (dispenses dispenser137 ingredient293)
  (dispenses dispenser354 ingredient394)
  (clean shaker181)
  (clean shot137)
  (empty shaker181)
  (empty shot137)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker181 l0)
  (shaker-level shaker181 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient394)
  (cocktail-part2 cocktail1 ingredient122)
)
 (:goal
  (and
      (contains shot137 cocktail1)
)))
