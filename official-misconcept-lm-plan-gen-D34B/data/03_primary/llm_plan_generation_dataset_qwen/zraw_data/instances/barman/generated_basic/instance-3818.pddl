(define (problem prob)
 (:domain barman)
 (:objects 
      shaker241 - shaker
      left right - hand
      shot21 shot256 - shot
      ingredient477 ingredient29 ingredient115 - ingredient
      cocktail482 - cocktail
      dispenser51 dispenser175 dispenser343 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker241)
  (ontable shot21)
  (ontable shot256)
  (dispenses dispenser51 ingredient477)
  (dispenses dispenser175 ingredient29)
  (dispenses dispenser343 ingredient115)
  (clean shaker241)
  (clean shot21)
  (clean shot256)
  (empty shaker241)
  (empty shot21)
  (empty shot256)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker241 l0)
  (shaker-level shaker241 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail482 ingredient115)
  (cocktail-part2 cocktail482 ingredient29)
)
 (:goal
  (and
      (contains shot21 cocktail482)
)))
