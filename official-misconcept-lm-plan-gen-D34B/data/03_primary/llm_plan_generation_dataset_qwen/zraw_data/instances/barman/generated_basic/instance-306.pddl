(define (problem prob)
 (:domain barman)
 (:objects 
      shaker201 - shaker
      left right - hand
      shot113 shot468 - shot
      ingredient67 ingredient380 - ingredient
      cocktail1 - cocktail
      dispenser365 dispenser291 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker201)
  (ontable shot113)
  (ontable shot468)
  (dispenses dispenser365 ingredient67)
  (dispenses dispenser291 ingredient380)
  (clean shaker201)
  (clean shot113)
  (clean shot468)
  (empty shaker201)
  (empty shot113)
  (empty shot468)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker201 l0)
  (shaker-level shaker201 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient67)
  (cocktail-part2 cocktail1 ingredient380)
)
 (:goal
  (and
      (contains shot113 cocktail1)
)))
