(define (problem prob)
 (:domain barman)
 (:objects 
      shaker184 - shaker
      left right - hand
      shot152 shot102 shot446 - shot
      ingredient495 ingredient313 ingredient19 ingredient191 - ingredient
      cocktail187 - cocktail
      dispenser460 dispenser92 dispenser217 dispenser190 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker184)
  (ontable shot152)
  (ontable shot102)
  (ontable shot446)
  (dispenses dispenser460 ingredient495)
  (dispenses dispenser92 ingredient313)
  (dispenses dispenser217 ingredient19)
  (dispenses dispenser190 ingredient191)
  (clean shaker184)
  (clean shot152)
  (clean shot102)
  (clean shot446)
  (empty shaker184)
  (empty shot152)
  (empty shot102)
  (empty shot446)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker184 l0)
  (shaker-level shaker184 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail187 ingredient313)
  (cocktail-part2 cocktail187 ingredient191)
)
 (:goal
  (and
      (contains shot152 cocktail187)
      (contains shot102 ingredient19)
)))
