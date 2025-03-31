(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot275 shot15 shot69 - shot
      ingredient274 ingredient339 - ingredient
      cocktail290 - cocktail
      dispenser263 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot275)
  (ontable shot15)
  (ontable shot69)
  (dispenses dispenser263 ingredient274)
  (dispenses dispenser196 ingredient339)
  (clean shaker359)
  (clean shot275)
  (clean shot15)
  (clean shot69)
  (empty shaker359)
  (empty shot275)
  (empty shot15)
  (empty shot69)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail290 ingredient274)
  (cocktail-part2 cocktail290 ingredient339)
)
 (:goal
  (and
      (contains shot275 cocktail290)
      (contains shot15 cocktail290)
)))
