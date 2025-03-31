(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot173 shot197 - shot
      ingredient170 ingredient216 - ingredient
      cocktail280 - cocktail
      dispenser380 dispenser2 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot173)
  (ontable shot197)
  (dispenses dispenser380 ingredient170)
  (dispenses dispenser2 ingredient216)
  (clean shaker474)
  (clean shot173)
  (clean shot197)
  (empty shaker474)
  (empty shot173)
  (empty shot197)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail280 ingredient170)
  (cocktail-part2 cocktail280 ingredient216)
)
 (:goal
  (and
      (contains shot173 cocktail280)
)))
