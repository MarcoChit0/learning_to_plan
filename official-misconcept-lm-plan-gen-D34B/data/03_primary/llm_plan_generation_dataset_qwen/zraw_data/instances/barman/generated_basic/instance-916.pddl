(define (problem prob)
 (:domain barman)
 (:objects 
      shaker230 - shaker
      left right - hand
      shot433 shot317 - shot
      ingredient476 ingredient215 ingredient321 - ingredient
      cocktail1 - cocktail
      dispenser290 dispenser267 dispenser406 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker230)
  (ontable shot433)
  (ontable shot317)
  (dispenses dispenser290 ingredient476)
  (dispenses dispenser267 ingredient215)
  (dispenses dispenser406 ingredient321)
  (clean shaker230)
  (clean shot433)
  (clean shot317)
  (empty shaker230)
  (empty shot433)
  (empty shot317)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker230 l0)
  (shaker-level shaker230 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient476)
  (cocktail-part2 cocktail1 ingredient215)
)
 (:goal
  (and
      (contains shot433 cocktail1)
)))
