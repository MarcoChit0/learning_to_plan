(define (problem prob)
 (:domain barman)
 (:objects 
      shaker250 - shaker
      left right - hand
      shot151 - shot
      ingredient301 ingredient54 ingredient429 ingredient223 - ingredient
      cocktail64 - cocktail
      dispenser160 dispenser331 dispenser168 dispenser321 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker250)
  (ontable shot151)
  (dispenses dispenser160 ingredient301)
  (dispenses dispenser331 ingredient54)
  (dispenses dispenser168 ingredient429)
  (dispenses dispenser321 ingredient223)
  (clean shaker250)
  (clean shot151)
  (empty shaker250)
  (empty shot151)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker250 l0)
  (shaker-level shaker250 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail64 ingredient54)
  (cocktail-part2 cocktail64 ingredient429)
)
 (:goal
  (and
      (contains shot151 cocktail64)
)))
