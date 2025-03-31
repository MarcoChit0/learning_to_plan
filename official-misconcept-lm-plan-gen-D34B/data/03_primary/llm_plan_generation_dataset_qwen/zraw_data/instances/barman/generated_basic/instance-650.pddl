(define (problem prob)
 (:domain barman)
 (:objects 
      shaker384 - shaker
      left right - hand
      shot313 shot22 - shot
      ingredient418 ingredient240 ingredient107 - ingredient
      cocktail1 - cocktail
      dispenser37 dispenser365 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker384)
  (ontable shot313)
  (ontable shot22)
  (dispenses dispenser37 ingredient418)
  (dispenses dispenser365 ingredient240)
  (dispenses dispenser292 ingredient107)
  (clean shaker384)
  (clean shot313)
  (clean shot22)
  (empty shaker384)
  (empty shot313)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker384 l0)
  (shaker-level shaker384 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient240)
  (cocktail-part2 cocktail1 ingredient418)
)
 (:goal
  (and
      (contains shot313 cocktail1)
)))
