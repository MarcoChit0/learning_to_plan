(define (problem prob)
 (:domain barman)
 (:objects 
      shaker145 - shaker
      left right - hand
      shot113 - shot
      ingredient67 ingredient184 ingredient433 ingredient148 - ingredient
      cocktail1 - cocktail
      dispenser160 dispenser98 dispenser279 dispenser200 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker145)
  (ontable shot113)
  (dispenses dispenser160 ingredient67)
  (dispenses dispenser98 ingredient184)
  (dispenses dispenser279 ingredient433)
  (dispenses dispenser200 ingredient148)
  (clean shaker145)
  (clean shot113)
  (empty shaker145)
  (empty shot113)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker145 l0)
  (shaker-level shaker145 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient67)
  (cocktail-part2 cocktail1 ingredient184)
)
 (:goal
  (and
      (contains shot113 cocktail1)
)))
