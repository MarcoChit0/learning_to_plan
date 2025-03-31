(define (problem prob)
 (:domain barman)
 (:objects 
      shaker165 - shaker
      left right - hand
      shot148 - shot
      ingredient274 ingredient254 - ingredient
      cocktail220 - cocktail
      dispenser141 dispenser80 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker165)
  (ontable shot148)
  (dispenses dispenser141 ingredient274)
  (dispenses dispenser80 ingredient254)
  (clean shaker165)
  (clean shot148)
  (empty shaker165)
  (empty shot148)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker165 l0)
  (shaker-level shaker165 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail220 ingredient274)
  (cocktail-part2 cocktail220 ingredient254)
)
 (:goal
  (and
      (contains shot148 cocktail220)
)))
