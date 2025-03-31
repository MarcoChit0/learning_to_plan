(define (problem prob)
 (:domain barman)
 (:objects 
      shaker392 - shaker
      left right - hand
      shot442 shot26 - shot
      ingredient304 ingredient286 ingredient420 ingredient332 - ingredient
      cocktail120 - cocktail
      dispenser145 dispenser327 dispenser78 dispenser388 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker392)
  (ontable shot442)
  (ontable shot26)
  (dispenses dispenser145 ingredient304)
  (dispenses dispenser327 ingredient286)
  (dispenses dispenser78 ingredient420)
  (dispenses dispenser388 ingredient332)
  (clean shaker392)
  (clean shot442)
  (clean shot26)
  (empty shaker392)
  (empty shot442)
  (empty shot26)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker392 l0)
  (shaker-level shaker392 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail120 ingredient332)
  (cocktail-part2 cocktail120 ingredient420)
)
 (:goal
  (and
      (contains shot442 cocktail120)
)))
