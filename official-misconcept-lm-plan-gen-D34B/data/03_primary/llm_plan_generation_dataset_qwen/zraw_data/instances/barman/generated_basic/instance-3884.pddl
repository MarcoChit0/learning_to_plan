(define (problem prob)
 (:domain barman)
 (:objects 
      shaker225 - shaker
      left right - hand
      shot178 shot350 - shot
      ingredient238 ingredient274 ingredient136 - ingredient
      cocktail307 - cocktail
      dispenser356 dispenser260 dispenser63 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker225)
  (ontable shot178)
  (ontable shot350)
  (dispenses dispenser356 ingredient238)
  (dispenses dispenser260 ingredient274)
  (dispenses dispenser63 ingredient136)
  (clean shaker225)
  (clean shot178)
  (clean shot350)
  (empty shaker225)
  (empty shot178)
  (empty shot350)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker225 l0)
  (shaker-level shaker225 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail307 ingredient136)
  (cocktail-part2 cocktail307 ingredient274)
)
 (:goal
  (and
      (contains shot178 cocktail307)
)))
