(define (problem prob)
 (:domain barman)
 (:objects 
      shaker205 - shaker
      left right - hand
      shot317 - shot
      ingredient32 ingredient497 ingredient348 - ingredient
      cocktail450 - cocktail
      dispenser4 dispenser207 dispenser90 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker205)
  (ontable shot317)
  (dispenses dispenser4 ingredient32)
  (dispenses dispenser207 ingredient497)
  (dispenses dispenser90 ingredient348)
  (clean shaker205)
  (clean shot317)
  (empty shaker205)
  (empty shot317)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker205 l0)
  (shaker-level shaker205 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail450 ingredient348)
  (cocktail-part2 cocktail450 ingredient32)
)
 (:goal
  (and
      (contains shot317 cocktail450)
)))
