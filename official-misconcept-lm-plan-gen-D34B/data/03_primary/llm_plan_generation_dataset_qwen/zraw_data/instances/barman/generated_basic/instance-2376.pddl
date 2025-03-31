(define (problem prob)
 (:domain barman)
 (:objects 
      shaker432 - shaker
      left right - hand
      shot490 - shot
      ingredient188 ingredient256 ingredient270 ingredient315 - ingredient
      cocktail193 - cocktail
      dispenser1 dispenser337 dispenser377 dispenser330 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker432)
  (ontable shot490)
  (dispenses dispenser1 ingredient188)
  (dispenses dispenser337 ingredient256)
  (dispenses dispenser377 ingredient270)
  (dispenses dispenser330 ingredient315)
  (clean shaker432)
  (clean shot490)
  (empty shaker432)
  (empty shot490)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker432 l0)
  (shaker-level shaker432 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail193 ingredient270)
  (cocktail-part2 cocktail193 ingredient188)
)
 (:goal
  (and
      (contains shot490 cocktail193)
)))
