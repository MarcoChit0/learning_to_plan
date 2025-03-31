(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot437 - shot
      ingredient183 ingredient178 ingredient135 ingredient425 - ingredient
      cocktail243 - cocktail
      dispenser264 dispenser249 dispenser284 dispenser79 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot437)
  (dispenses dispenser264 ingredient183)
  (dispenses dispenser249 ingredient178)
  (dispenses dispenser284 ingredient135)
  (dispenses dispenser79 ingredient425)
  (clean shaker359)
  (clean shot437)
  (empty shaker359)
  (empty shot437)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail243 ingredient183)
  (cocktail-part2 cocktail243 ingredient135)
)
 (:goal
  (and
      (contains shot437 cocktail243)
)))
