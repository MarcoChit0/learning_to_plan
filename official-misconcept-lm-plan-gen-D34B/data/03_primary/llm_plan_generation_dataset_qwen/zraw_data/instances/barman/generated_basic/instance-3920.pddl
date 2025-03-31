(define (problem prob)
 (:domain barman)
 (:objects 
      shaker318 - shaker
      left right - hand
      shot459 - shot
      ingredient274 ingredient38 ingredient126 ingredient294 - ingredient
      cocktail334 - cocktail
      dispenser56 dispenser211 dispenser328 dispenser110 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker318)
  (ontable shot459)
  (dispenses dispenser56 ingredient274)
  (dispenses dispenser211 ingredient38)
  (dispenses dispenser328 ingredient126)
  (dispenses dispenser110 ingredient294)
  (clean shaker318)
  (clean shot459)
  (empty shaker318)
  (empty shot459)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker318 l0)
  (shaker-level shaker318 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail334 ingredient274)
  (cocktail-part2 cocktail334 ingredient38)
)
 (:goal
  (and
      (contains shot459 cocktail334)
)))
