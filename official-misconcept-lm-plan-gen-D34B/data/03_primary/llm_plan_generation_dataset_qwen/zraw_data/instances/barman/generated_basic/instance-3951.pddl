(define (problem prob)
 (:domain barman)
 (:objects 
      shaker259 - shaker
      left right - hand
      shot454 shot45 - shot
      ingredient152 ingredient12 - ingredient
      cocktail320 - cocktail
      dispenser109 dispenser18 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker259)
  (ontable shot454)
  (ontable shot45)
  (dispenses dispenser109 ingredient152)
  (dispenses dispenser18 ingredient12)
  (clean shaker259)
  (clean shot454)
  (clean shot45)
  (empty shaker259)
  (empty shot454)
  (empty shot45)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker259 l0)
  (shaker-level shaker259 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail320 ingredient152)
  (cocktail-part2 cocktail320 ingredient12)
)
 (:goal
  (and
      (contains shot454 cocktail320)
)))
