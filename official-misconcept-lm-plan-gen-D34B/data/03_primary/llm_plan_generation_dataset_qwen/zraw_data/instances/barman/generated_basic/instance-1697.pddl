(define (problem prob)
 (:domain barman)
 (:objects 
      shaker449 - shaker
      left right - hand
      shot94 - shot
      ingredient141 ingredient485 ingredient51 ingredient258 - ingredient
      cocktail411 - cocktail
      dispenser259 dispenser365 dispenser8 dispenser213 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker449)
  (ontable shot94)
  (dispenses dispenser259 ingredient141)
  (dispenses dispenser365 ingredient485)
  (dispenses dispenser8 ingredient51)
  (dispenses dispenser213 ingredient258)
  (clean shaker449)
  (clean shot94)
  (empty shaker449)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker449 l0)
  (shaker-level shaker449 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail411 ingredient258)
  (cocktail-part2 cocktail411 ingredient141)
)
 (:goal
  (and
      (contains shot94 cocktail411)
)))
