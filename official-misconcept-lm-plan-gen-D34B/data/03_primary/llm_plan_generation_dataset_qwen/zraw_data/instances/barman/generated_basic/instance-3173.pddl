(define (problem prob)
 (:domain barman)
 (:objects 
      shaker396 - shaker
      left right - hand
      shot152 - shot
      ingredient254 ingredient163 ingredient49 - ingredient
      cocktail436 - cocktail
      dispenser17 dispenser472 dispenser152 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker396)
  (ontable shot152)
  (dispenses dispenser17 ingredient254)
  (dispenses dispenser472 ingredient163)
  (dispenses dispenser152 ingredient49)
  (clean shaker396)
  (clean shot152)
  (empty shaker396)
  (empty shot152)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker396 l0)
  (shaker-level shaker396 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail436 ingredient163)
  (cocktail-part2 cocktail436 ingredient49)
)
 (:goal
  (and
      (contains shot152 cocktail436)
)))
