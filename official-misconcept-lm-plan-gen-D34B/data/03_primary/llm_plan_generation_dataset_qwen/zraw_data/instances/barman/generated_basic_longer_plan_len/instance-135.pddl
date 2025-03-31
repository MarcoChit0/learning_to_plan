(define (problem prob)
 (:domain barman)
 (:objects 
      shaker437 - shaker
      left right - hand
      shot60 shot63 - shot
      ingredient274 ingredient286 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser349 dispenser361 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker437)
  (ontable shot60)
  (ontable shot63)
  (dispenses dispenser349 ingredient274)
  (dispenses dispenser361 ingredient286)
  (clean shaker437)
  (clean shot60)
  (clean shot63)
  (empty shaker437)
  (empty shot60)
  (empty shot63)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker437 l0)
  (shaker-level shaker437 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient286)
  (cocktail-part2 cocktail1 ingredient274)
  (cocktail-part1 cocktail2 ingredient274)
  (cocktail-part2 cocktail2 ingredient286)
)
 (:goal
  (and
      (contains shot60 cocktail1)
      (contains shot63 cocktail2)
)))
