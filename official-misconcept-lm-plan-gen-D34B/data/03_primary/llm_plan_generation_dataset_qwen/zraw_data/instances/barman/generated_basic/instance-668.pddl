(define (problem prob)
 (:domain barman)
 (:objects 
      shaker289 - shaker
      left right - hand
      shot100 shot68 - shot
      ingredient274 ingredient449 - ingredient
      cocktail1 - cocktail
      dispenser456 dispenser120 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker289)
  (ontable shot100)
  (ontable shot68)
  (dispenses dispenser456 ingredient274)
  (dispenses dispenser120 ingredient449)
  (clean shaker289)
  (clean shot100)
  (clean shot68)
  (empty shaker289)
  (empty shot100)
  (empty shot68)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker289 l0)
  (shaker-level shaker289 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient449)
  (cocktail-part2 cocktail1 ingredient274)
)
 (:goal
  (and
      (contains shot100 cocktail1)
)))
