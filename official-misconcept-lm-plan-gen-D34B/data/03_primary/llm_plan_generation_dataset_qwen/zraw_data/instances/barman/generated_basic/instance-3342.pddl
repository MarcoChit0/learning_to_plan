(define (problem prob)
 (:domain barman)
 (:objects 
      shaker94 - shaker
      left right - hand
      shot298 shot224 - shot
      ingredient274 ingredient84 - ingredient
      cocktail432 - cocktail
      dispenser175 dispenser399 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker94)
  (ontable shot298)
  (ontable shot224)
  (dispenses dispenser175 ingredient274)
  (dispenses dispenser399 ingredient84)
  (clean shaker94)
  (clean shot298)
  (clean shot224)
  (empty shaker94)
  (empty shot298)
  (empty shot224)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker94 l0)
  (shaker-level shaker94 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail432 ingredient274)
  (cocktail-part2 cocktail432 ingredient84)
)
 (:goal
  (and
      (contains shot298 cocktail432)
)))
