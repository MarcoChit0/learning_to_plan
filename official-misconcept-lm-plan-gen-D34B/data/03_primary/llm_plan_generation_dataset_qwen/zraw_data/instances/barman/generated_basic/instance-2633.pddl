(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot97 shot162 - shot
      ingredient58 ingredient334 ingredient96 - ingredient
      cocktail236 - cocktail
      dispenser62 dispenser485 dispenser426 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot97)
  (ontable shot162)
  (dispenses dispenser62 ingredient58)
  (dispenses dispenser485 ingredient334)
  (dispenses dispenser426 ingredient96)
  (clean shaker359)
  (clean shot97)
  (clean shot162)
  (empty shaker359)
  (empty shot97)
  (empty shot162)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail236 ingredient334)
  (cocktail-part2 cocktail236 ingredient58)
)
 (:goal
  (and
      (contains shot97 cocktail236)
)))
