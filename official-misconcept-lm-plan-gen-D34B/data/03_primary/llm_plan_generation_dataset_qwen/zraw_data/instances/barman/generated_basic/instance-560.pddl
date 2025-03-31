(define (problem prob)
 (:domain barman)
 (:objects 
      shaker172 - shaker
      left right - hand
      shot198 shot221 - shot
      ingredient304 ingredient484 ingredient274 - ingredient
      cocktail1 - cocktail
      dispenser268 dispenser135 dispenser414 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker172)
  (ontable shot198)
  (ontable shot221)
  (dispenses dispenser268 ingredient304)
  (dispenses dispenser135 ingredient484)
  (dispenses dispenser414 ingredient274)
  (clean shaker172)
  (clean shot198)
  (clean shot221)
  (empty shaker172)
  (empty shot198)
  (empty shot221)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker172 l0)
  (shaker-level shaker172 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient304)
  (cocktail-part2 cocktail1 ingredient484)
)
 (:goal
  (and
      (contains shot198 cocktail1)
)))
