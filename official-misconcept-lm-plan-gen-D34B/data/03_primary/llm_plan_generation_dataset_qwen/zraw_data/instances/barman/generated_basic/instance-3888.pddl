(define (problem prob)
 (:domain barman)
 (:objects 
      shaker211 - shaker
      left right - hand
      shot6 shot194 - shot
      ingredient93 ingredient43 ingredient98 - ingredient
      cocktail423 - cocktail
      dispenser311 dispenser278 dispenser26 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker211)
  (ontable shot6)
  (ontable shot194)
  (dispenses dispenser311 ingredient93)
  (dispenses dispenser278 ingredient43)
  (dispenses dispenser26 ingredient98)
  (clean shaker211)
  (clean shot6)
  (clean shot194)
  (empty shaker211)
  (empty shot6)
  (empty shot194)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker211 l0)
  (shaker-level shaker211 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail423 ingredient43)
  (cocktail-part2 cocktail423 ingredient93)
)
 (:goal
  (and
      (contains shot6 cocktail423)
)))
