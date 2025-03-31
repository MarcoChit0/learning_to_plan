(define (problem prob)
 (:domain barman)
 (:objects 
      shaker428 - shaker
      left right - hand
      shot472 shot245 - shot
      ingredient421 ingredient221 ingredient218 - ingredient
      cocktail256 - cocktail
      dispenser429 dispenser21 dispenser406 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker428)
  (ontable shot472)
  (ontable shot245)
  (dispenses dispenser429 ingredient421)
  (dispenses dispenser21 ingredient221)
  (dispenses dispenser406 ingredient218)
  (clean shaker428)
  (clean shot472)
  (clean shot245)
  (empty shaker428)
  (empty shot472)
  (empty shot245)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker428 l0)
  (shaker-level shaker428 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail256 ingredient218)
  (cocktail-part2 cocktail256 ingredient221)
)
 (:goal
  (and
      (contains shot472 cocktail256)
)))
