(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot287 - shot
      ingredient383 ingredient153 - ingredient
      cocktail467 - cocktail
      dispenser10 dispenser133 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot287)
  (dispenses dispenser10 ingredient383)
  (dispenses dispenser133 ingredient153)
  (clean shaker359)
  (clean shot287)
  (empty shaker359)
  (empty shot287)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail467 ingredient153)
  (cocktail-part2 cocktail467 ingredient383)
)
 (:goal
  (and
      (contains shot287 cocktail467)
)))
