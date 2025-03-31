(define (problem prob)
 (:domain barman)
 (:objects 
      shaker355 - shaker
      left right - hand
      shot91 shot453 - shot
      ingredient86 ingredient156 - ingredient
      cocktail1 - cocktail
      dispenser342 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker355)
  (ontable shot91)
  (ontable shot453)
  (dispenses dispenser342 ingredient86)
  (dispenses dispenser370 ingredient156)
  (clean shaker355)
  (clean shot91)
  (clean shot453)
  (empty shaker355)
  (empty shot91)
  (empty shot453)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker355 l0)
  (shaker-level shaker355 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient156)
  (cocktail-part2 cocktail1 ingredient86)
)
 (:goal
  (and
      (contains shot91 cocktail1)
)))
