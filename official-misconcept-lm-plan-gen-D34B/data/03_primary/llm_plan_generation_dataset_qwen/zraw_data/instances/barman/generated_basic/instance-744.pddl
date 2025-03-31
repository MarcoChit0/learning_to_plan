(define (problem prob)
 (:domain barman)
 (:objects 
      shaker384 - shaker
      left right - hand
      shot81 shot257 - shot
      ingredient186 ingredient419 ingredient365 - ingredient
      cocktail1 - cocktail
      dispenser123 dispenser364 dispenser200 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker384)
  (ontable shot81)
  (ontable shot257)
  (dispenses dispenser123 ingredient186)
  (dispenses dispenser364 ingredient419)
  (dispenses dispenser200 ingredient365)
  (clean shaker384)
  (clean shot81)
  (clean shot257)
  (empty shaker384)
  (empty shot81)
  (empty shot257)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker384 l0)
  (shaker-level shaker384 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient419)
  (cocktail-part2 cocktail1 ingredient186)
)
 (:goal
  (and
      (contains shot81 cocktail1)
)))
