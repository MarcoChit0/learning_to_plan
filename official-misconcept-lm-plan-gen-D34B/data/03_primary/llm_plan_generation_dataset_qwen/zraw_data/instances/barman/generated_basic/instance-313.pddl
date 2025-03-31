(define (problem prob)
 (:domain barman)
 (:objects 
      shaker477 - shaker
      left right - hand
      shot321 shot270 - shot
      ingredient346 ingredient482 - ingredient
      cocktail1 - cocktail
      dispenser228 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker477)
  (ontable shot321)
  (ontable shot270)
  (dispenses dispenser228 ingredient346)
  (dispenses dispenser290 ingredient482)
  (clean shaker477)
  (clean shot321)
  (clean shot270)
  (empty shaker477)
  (empty shot321)
  (empty shot270)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker477 l0)
  (shaker-level shaker477 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient346)
  (cocktail-part2 cocktail1 ingredient482)
)
 (:goal
  (and
      (contains shot321 cocktail1)
)))
