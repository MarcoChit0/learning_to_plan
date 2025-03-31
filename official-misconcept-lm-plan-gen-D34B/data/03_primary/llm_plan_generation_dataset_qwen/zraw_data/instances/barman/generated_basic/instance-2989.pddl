(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot311 shot417 - shot
      ingredient396 ingredient274 ingredient445 ingredient167 - ingredient
      cocktail188 - cocktail
      dispenser111 dispenser189 dispenser64 dispenser205 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot311)
  (ontable shot417)
  (dispenses dispenser111 ingredient396)
  (dispenses dispenser189 ingredient274)
  (dispenses dispenser64 ingredient445)
  (dispenses dispenser205 ingredient167)
  (clean shaker468)
  (clean shot311)
  (clean shot417)
  (empty shaker468)
  (empty shot311)
  (empty shot417)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail188 ingredient274)
  (cocktail-part2 cocktail188 ingredient167)
)
 (:goal
  (and
      (contains shot311 cocktail188)
)))
