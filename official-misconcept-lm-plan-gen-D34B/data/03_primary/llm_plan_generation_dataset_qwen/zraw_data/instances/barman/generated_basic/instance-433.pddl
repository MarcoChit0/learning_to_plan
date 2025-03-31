(define (problem prob)
 (:domain barman)
 (:objects 
      shaker37 - shaker
      left right - hand
      shot449 - shot
      ingredient367 ingredient172 - ingredient
      cocktail1 - cocktail
      dispenser311 dispenser316 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker37)
  (ontable shot449)
  (dispenses dispenser311 ingredient367)
  (dispenses dispenser316 ingredient172)
  (clean shaker37)
  (clean shot449)
  (empty shaker37)
  (empty shot449)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker37 l0)
  (shaker-level shaker37 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient172)
  (cocktail-part2 cocktail1 ingredient367)
)
 (:goal
  (and
      (contains shot449 cocktail1)
)))
