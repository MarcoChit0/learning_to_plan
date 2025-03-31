(define (problem prob)
 (:domain barman)
 (:objects 
      shaker449 - shaker
      left right - hand
      shot177 shot146 - shot
      ingredient491 ingredient480 - ingredient
      cocktail1 - cocktail
      dispenser197 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker449)
  (ontable shot177)
  (ontable shot146)
  (dispenses dispenser197 ingredient491)
  (dispenses dispenser292 ingredient480)
  (clean shaker449)
  (clean shot177)
  (clean shot146)
  (empty shaker449)
  (empty shot177)
  (empty shot146)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker449 l0)
  (shaker-level shaker449 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient491)
  (cocktail-part2 cocktail1 ingredient480)
)
 (:goal
  (and
      (contains shot177 cocktail1)
)))
