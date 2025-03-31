(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot103 shot176 - shot
      ingredient78 ingredient350 ingredient412 - ingredient
      cocktail180 - cocktail
      dispenser244 dispenser285 dispenser236 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot103)
  (ontable shot176)
  (dispenses dispenser244 ingredient78)
  (dispenses dispenser285 ingredient350)
  (dispenses dispenser236 ingredient412)
  (clean shaker34)
  (clean shot103)
  (clean shot176)
  (empty shaker34)
  (empty shot103)
  (empty shot176)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail180 ingredient78)
  (cocktail-part2 cocktail180 ingredient412)
)
 (:goal
  (and
      (contains shot103 cocktail180)
)))
