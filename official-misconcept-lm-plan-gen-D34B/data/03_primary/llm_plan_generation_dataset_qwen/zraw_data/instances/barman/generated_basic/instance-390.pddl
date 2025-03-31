(define (problem prob)
 (:domain barman)
 (:objects 
      shaker238 - shaker
      left right - hand
      shot237 shot231 shot418 - shot
      ingredient417 ingredient274 ingredient165 - ingredient
      cocktail1 - cocktail
      dispenser35 dispenser339 dispenser34 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker238)
  (ontable shot237)
  (ontable shot231)
  (ontable shot418)
  (dispenses dispenser35 ingredient417)
  (dispenses dispenser339 ingredient274)
  (dispenses dispenser34 ingredient165)
  (clean shaker238)
  (clean shot237)
  (clean shot231)
  (clean shot418)
  (empty shaker238)
  (empty shot237)
  (empty shot231)
  (empty shot418)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker238 l0)
  (shaker-level shaker238 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient274)
  (cocktail-part2 cocktail1 ingredient417)
)
 (:goal
  (and
      (contains shot237 cocktail1)
      (contains shot231 ingredient274)
)))
