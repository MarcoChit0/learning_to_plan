(define (problem prob)
 (:domain barman)
 (:objects 
      shaker353 - shaker
      left right - hand
      shot208 shot126 shot53 - shot
      ingredient184 ingredient89 - ingredient
      cocktail350 - cocktail
      dispenser335 dispenser291 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker353)
  (ontable shot208)
  (ontable shot126)
  (ontable shot53)
  (dispenses dispenser335 ingredient184)
  (dispenses dispenser291 ingredient89)
  (clean shaker353)
  (clean shot208)
  (clean shot126)
  (clean shot53)
  (empty shaker353)
  (empty shot208)
  (empty shot126)
  (empty shot53)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker353 l0)
  (shaker-level shaker353 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail350 ingredient89)
  (cocktail-part2 cocktail350 ingredient184)
)
 (:goal
  (and
      (contains shot208 cocktail350)
      (contains shot126 ingredient184)
)))
