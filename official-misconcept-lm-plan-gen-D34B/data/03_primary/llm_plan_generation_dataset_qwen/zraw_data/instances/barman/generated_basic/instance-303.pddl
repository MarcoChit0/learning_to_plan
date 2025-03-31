(define (problem prob)
 (:domain barman)
 (:objects 
      shaker22 - shaker
      left right - hand
      shot32 shot499 - shot
      ingredient110 ingredient283 - ingredient
      cocktail1 - cocktail
      dispenser247 dispenser451 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker22)
  (ontable shot32)
  (ontable shot499)
  (dispenses dispenser247 ingredient110)
  (dispenses dispenser451 ingredient283)
  (clean shaker22)
  (clean shot32)
  (clean shot499)
  (empty shaker22)
  (empty shot32)
  (empty shot499)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker22 l0)
  (shaker-level shaker22 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient283)
  (cocktail-part2 cocktail1 ingredient110)
)
 (:goal
  (and
      (contains shot32 cocktail1)
)))
