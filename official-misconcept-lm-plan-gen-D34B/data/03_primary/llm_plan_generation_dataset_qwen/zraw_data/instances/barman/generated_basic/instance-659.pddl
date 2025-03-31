(define (problem prob)
 (:domain barman)
 (:objects 
      shaker304 - shaker
      left right - hand
      shot374 shot232 - shot
      ingredient35 ingredient393 ingredient126 - ingredient
      cocktail1 - cocktail
      dispenser437 dispenser262 dispenser468 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker304)
  (ontable shot374)
  (ontable shot232)
  (dispenses dispenser437 ingredient35)
  (dispenses dispenser262 ingredient393)
  (dispenses dispenser468 ingredient126)
  (clean shaker304)
  (clean shot374)
  (clean shot232)
  (empty shaker304)
  (empty shot374)
  (empty shot232)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker304 l0)
  (shaker-level shaker304 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient126)
  (cocktail-part2 cocktail1 ingredient35)
)
 (:goal
  (and
      (contains shot374 cocktail1)
)))
