(define (problem prob)
 (:domain barman)
 (:objects 
      shaker399 - shaker
      left right - hand
      shot184 shot215 - shot
      ingredient189 ingredient50 ingredient431 ingredient274 - ingredient
      cocktail9 - cocktail
      dispenser91 dispenser202 dispenser329 dispenser133 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker399)
  (ontable shot184)
  (ontable shot215)
  (dispenses dispenser91 ingredient189)
  (dispenses dispenser202 ingredient50)
  (dispenses dispenser329 ingredient431)
  (dispenses dispenser133 ingredient274)
  (clean shaker399)
  (clean shot184)
  (clean shot215)
  (empty shaker399)
  (empty shot184)
  (empty shot215)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker399 l0)
  (shaker-level shaker399 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail9 ingredient274)
  (cocktail-part2 cocktail9 ingredient189)
)
 (:goal
  (and
      (contains shot184 cocktail9)
)))
