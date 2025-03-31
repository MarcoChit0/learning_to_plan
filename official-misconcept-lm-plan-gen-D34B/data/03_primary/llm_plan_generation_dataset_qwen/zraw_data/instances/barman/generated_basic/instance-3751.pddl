(define (problem prob)
 (:domain barman)
 (:objects 
      shaker123 - shaker
      left right - hand
      shot165 shot74 shot187 - shot
      ingredient382 ingredient379 - ingredient
      cocktail5 - cocktail
      dispenser393 dispenser55 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker123)
  (ontable shot165)
  (ontable shot74)
  (ontable shot187)
  (dispenses dispenser393 ingredient382)
  (dispenses dispenser55 ingredient379)
  (clean shaker123)
  (clean shot165)
  (clean shot74)
  (clean shot187)
  (empty shaker123)
  (empty shot165)
  (empty shot74)
  (empty shot187)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker123 l0)
  (shaker-level shaker123 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail5 ingredient379)
  (cocktail-part2 cocktail5 ingredient382)
)
 (:goal
  (and
      (contains shot165 cocktail5)
      (contains shot74 cocktail5)
)))
