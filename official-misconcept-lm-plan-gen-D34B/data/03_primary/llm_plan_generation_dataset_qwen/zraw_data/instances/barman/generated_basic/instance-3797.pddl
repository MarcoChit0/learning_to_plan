(define (problem prob)
 (:domain barman)
 (:objects 
      shaker5 - shaker
      left right - hand
      shot401 - shot
      ingredient422 ingredient274 ingredient243 ingredient218 - ingredient
      cocktail463 - cocktail
      dispenser448 dispenser77 dispenser265 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker5)
  (ontable shot401)
  (dispenses dispenser448 ingredient422)
  (dispenses dispenser77 ingredient274)
  (dispenses dispenser265 ingredient243)
  (dispenses dispenser92 ingredient218)
  (clean shaker5)
  (clean shot401)
  (empty shaker5)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker5 l0)
  (shaker-level shaker5 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail463 ingredient422)
  (cocktail-part2 cocktail463 ingredient218)
)
 (:goal
  (and
      (contains shot401 cocktail463)
)))
