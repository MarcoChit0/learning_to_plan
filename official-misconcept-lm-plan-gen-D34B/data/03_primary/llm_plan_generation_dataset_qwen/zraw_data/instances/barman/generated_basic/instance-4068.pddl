(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot237 shot366 - shot
      ingredient423 ingredient443 - ingredient
      cocktail382 - cocktail
      dispenser431 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot237)
  (ontable shot366)
  (dispenses dispenser431 ingredient423)
  (dispenses dispenser92 ingredient443)
  (clean shaker359)
  (clean shot237)
  (clean shot366)
  (empty shaker359)
  (empty shot237)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail382 ingredient423)
  (cocktail-part2 cocktail382 ingredient443)
)
 (:goal
  (and
      (contains shot237 cocktail382)
)))
