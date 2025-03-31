(define (problem prob)
 (:domain barman)
 (:objects 
      shaker220 - shaker
      left right - hand
      shot305 shot172 - shot
      ingredient416 ingredient47 ingredient306 - ingredient
      cocktail313 - cocktail
      dispenser401 dispenser318 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker220)
  (ontable shot305)
  (ontable shot172)
  (dispenses dispenser401 ingredient416)
  (dispenses dispenser318 ingredient47)
  (dispenses dispenser149 ingredient306)
  (clean shaker220)
  (clean shot305)
  (clean shot172)
  (empty shaker220)
  (empty shot305)
  (empty shot172)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker220 l0)
  (shaker-level shaker220 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail313 ingredient47)
  (cocktail-part2 cocktail313 ingredient416)
)
 (:goal
  (and
      (contains shot305 cocktail313)
)))
