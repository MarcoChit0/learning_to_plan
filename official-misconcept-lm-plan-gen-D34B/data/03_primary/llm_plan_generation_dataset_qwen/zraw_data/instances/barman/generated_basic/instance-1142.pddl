(define (problem prob)
 (:domain barman)
 (:objects 
      shaker267 - shaker
      left right - hand
      shot33 shot211 - shot
      ingredient208 ingredient60 ingredient41 - ingredient
      cocktail1 - cocktail
      dispenser484 dispenser184 dispenser346 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker267)
  (ontable shot33)
  (ontable shot211)
  (dispenses dispenser484 ingredient208)
  (dispenses dispenser184 ingredient60)
  (dispenses dispenser346 ingredient41)
  (clean shaker267)
  (clean shot33)
  (clean shot211)
  (empty shaker267)
  (empty shot33)
  (empty shot211)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker267 l0)
  (shaker-level shaker267 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient41)
  (cocktail-part2 cocktail1 ingredient60)
)
 (:goal
  (and
      (contains shot33 cocktail1)
)))
