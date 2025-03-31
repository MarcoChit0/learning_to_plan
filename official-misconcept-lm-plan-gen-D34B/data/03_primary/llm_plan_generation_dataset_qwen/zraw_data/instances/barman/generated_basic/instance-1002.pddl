(define (problem prob)
 (:domain barman)
 (:objects 
      shaker395 - shaker
      left right - hand
      shot179 shot31 - shot
      ingredient494 ingredient265 - ingredient
      cocktail1 - cocktail
      dispenser414 dispenser445 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker395)
  (ontable shot179)
  (ontable shot31)
  (dispenses dispenser414 ingredient494)
  (dispenses dispenser445 ingredient265)
  (clean shaker395)
  (clean shot179)
  (clean shot31)
  (empty shaker395)
  (empty shot179)
  (empty shot31)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker395 l0)
  (shaker-level shaker395 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient494)
  (cocktail-part2 cocktail1 ingredient265)
)
 (:goal
  (and
      (contains shot179 cocktail1)
)))
