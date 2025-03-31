(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot186 shot189 - shot
      ingredient490 ingredient485 - ingredient
      cocktail499 - cocktail
      dispenser206 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot186)
  (ontable shot189)
  (dispenses dispenser206 ingredient490)
  (dispenses dispenser150 ingredient485)
  (clean shaker359)
  (clean shot186)
  (clean shot189)
  (empty shaker359)
  (empty shot186)
  (empty shot189)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail499 ingredient490)
  (cocktail-part2 cocktail499 ingredient485)
)
 (:goal
  (and
      (contains shot186 cocktail499)
)))
