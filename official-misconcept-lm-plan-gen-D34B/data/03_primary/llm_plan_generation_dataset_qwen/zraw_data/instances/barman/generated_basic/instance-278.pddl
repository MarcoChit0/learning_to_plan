(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot291 shot318 - shot
      ingredient221 ingredient374 ingredient180 - ingredient
      cocktail1 - cocktail
      dispenser75 dispenser160 dispenser405 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot291)
  (ontable shot318)
  (dispenses dispenser75 ingredient221)
  (dispenses dispenser160 ingredient374)
  (dispenses dispenser405 ingredient180)
  (clean shaker39)
  (clean shot291)
  (clean shot318)
  (empty shaker39)
  (empty shot291)
  (empty shot318)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient374)
  (cocktail-part2 cocktail1 ingredient221)
)
 (:goal
  (and
      (contains shot291 cocktail1)
)))
