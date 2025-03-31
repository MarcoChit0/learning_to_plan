(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot69 - shot
      ingredient163 ingredient399 ingredient247 - ingredient
      cocktail1 - cocktail
      dispenser40 dispenser37 dispenser456 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot69)
  (dispenses dispenser40 ingredient163)
  (dispenses dispenser37 ingredient399)
  (dispenses dispenser456 ingredient247)
  (clean shaker498)
  (clean shot69)
  (empty shaker498)
  (empty shot69)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient247)
  (cocktail-part2 cocktail1 ingredient163)
)
 (:goal
  (and
      (contains shot69 cocktail1)
)))
