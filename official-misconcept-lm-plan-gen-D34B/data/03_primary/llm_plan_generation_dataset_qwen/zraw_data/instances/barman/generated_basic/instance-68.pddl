(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot461 shot409 - shot
      ingredient28 ingredient231 ingredient332 - ingredient
      cocktail1 - cocktail
      dispenser408 dispenser244 dispenser93 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot461)
  (ontable shot409)
  (dispenses dispenser408 ingredient28)
  (dispenses dispenser244 ingredient231)
  (dispenses dispenser93 ingredient332)
  (clean shaker375)
  (clean shot461)
  (clean shot409)
  (empty shaker375)
  (empty shot461)
  (empty shot409)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient332)
  (cocktail-part2 cocktail1 ingredient28)
)
 (:goal
  (and
      (contains shot461 cocktail1)
)))
