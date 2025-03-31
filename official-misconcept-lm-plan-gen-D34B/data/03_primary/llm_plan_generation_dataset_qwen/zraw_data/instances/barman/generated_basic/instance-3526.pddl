(define (problem prob)
 (:domain barman)
 (:objects 
      shaker56 - shaker
      left right - hand
      shot127 shot173 - shot
      ingredient309 ingredient347 ingredient405 ingredient85 - ingredient
      cocktail449 - cocktail
      dispenser199 dispenser67 dispenser105 dispenser360 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker56)
  (ontable shot127)
  (ontable shot173)
  (dispenses dispenser199 ingredient309)
  (dispenses dispenser67 ingredient347)
  (dispenses dispenser105 ingredient405)
  (dispenses dispenser360 ingredient85)
  (clean shaker56)
  (clean shot127)
  (clean shot173)
  (empty shaker56)
  (empty shot127)
  (empty shot173)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker56 l0)
  (shaker-level shaker56 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail449 ingredient309)
  (cocktail-part2 cocktail449 ingredient347)
)
 (:goal
  (and
      (contains shot127 cocktail449)
)))
