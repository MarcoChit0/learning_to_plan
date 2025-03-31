(define (problem prob)
 (:domain barman)
 (:objects 
      shaker367 - shaker
      left right - hand
      shot471 shot366 - shot
      ingredient235 ingredient450 ingredient274 - ingredient
      cocktail1 - cocktail
      dispenser490 dispenser108 dispenser158 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker367)
  (ontable shot471)
  (ontable shot366)
  (dispenses dispenser490 ingredient235)
  (dispenses dispenser108 ingredient450)
  (dispenses dispenser158 ingredient274)
  (clean shaker367)
  (clean shot471)
  (clean shot366)
  (empty shaker367)
  (empty shot471)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker367 l0)
  (shaker-level shaker367 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient450)
  (cocktail-part2 cocktail1 ingredient274)
)
 (:goal
  (and
      (contains shot471 cocktail1)
)))
