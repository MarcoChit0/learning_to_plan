(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot442 - shot
      ingredient337 ingredient117 - ingredient
      cocktail1 - cocktail
      dispenser226 dispenser379 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot442)
  (dispenses dispenser226 ingredient337)
  (dispenses dispenser379 ingredient117)
  (clean shaker359)
  (clean shot442)
  (empty shaker359)
  (empty shot442)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient117)
  (cocktail-part2 cocktail1 ingredient337)
)
 (:goal
  (and
      (contains shot442 cocktail1)
)))
