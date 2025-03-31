(define (problem prob)
 (:domain barman)
 (:objects 
      shaker484 - shaker
      left right - hand
      shot489 shot498 - shot
      ingredient281 ingredient310 - ingredient
      cocktail127 - cocktail
      dispenser321 dispenser168 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker484)
  (ontable shot489)
  (ontable shot498)
  (dispenses dispenser321 ingredient281)
  (dispenses dispenser168 ingredient310)
  (clean shaker484)
  (clean shot489)
  (clean shot498)
  (empty shaker484)
  (empty shot489)
  (empty shot498)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker484 l0)
  (shaker-level shaker484 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail127 ingredient310)
  (cocktail-part2 cocktail127 ingredient281)
)
 (:goal
  (and
      (contains shot489 cocktail127)
)))
