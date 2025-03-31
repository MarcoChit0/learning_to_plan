(define (problem prob)
 (:domain barman)
 (:objects 
      shaker65 - shaker
      left right - hand
      shot461 shot299 shot190 - shot
      ingredient260 ingredient451 - ingredient
      cocktail59 - cocktail
      dispenser394 dispenser449 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker65)
  (ontable shot461)
  (ontable shot299)
  (ontable shot190)
  (dispenses dispenser394 ingredient260)
  (dispenses dispenser449 ingredient451)
  (clean shaker65)
  (clean shot461)
  (clean shot299)
  (clean shot190)
  (empty shaker65)
  (empty shot461)
  (empty shot299)
  (empty shot190)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker65 l0)
  (shaker-level shaker65 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail59 ingredient260)
  (cocktail-part2 cocktail59 ingredient451)
)
 (:goal
  (and
      (contains shot461 cocktail59)
      (contains shot299 ingredient451)
)))
