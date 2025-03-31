(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot0 - shot
      ingredient139 ingredient398 ingredient114 - ingredient
      cocktail1 - cocktail
      dispenser37 dispenser76 dispenser58 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot0)
  (dispenses dispenser37 ingredient139)
  (dispenses dispenser76 ingredient398)
  (dispenses dispenser58 ingredient114)
  (clean shaker498)
  (clean shot0)
  (empty shaker498)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient114)
  (cocktail-part2 cocktail1 ingredient139)
)
 (:goal
  (and
      (contains shot0 cocktail1)
)))
