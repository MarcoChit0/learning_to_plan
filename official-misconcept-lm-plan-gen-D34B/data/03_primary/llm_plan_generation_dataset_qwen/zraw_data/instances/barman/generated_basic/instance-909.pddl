(define (problem prob)
 (:domain barman)
 (:objects 
      shaker318 - shaker
      left right - hand
      shot153 shot279 - shot
      ingredient418 ingredient73 ingredient468 - ingredient
      cocktail1 - cocktail
      dispenser49 dispenser304 dispenser394 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker318)
  (ontable shot153)
  (ontable shot279)
  (dispenses dispenser49 ingredient418)
  (dispenses dispenser304 ingredient73)
  (dispenses dispenser394 ingredient468)
  (clean shaker318)
  (clean shot153)
  (clean shot279)
  (empty shaker318)
  (empty shot153)
  (empty shot279)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker318 l0)
  (shaker-level shaker318 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient468)
  (cocktail-part2 cocktail1 ingredient73)
)
 (:goal
  (and
      (contains shot153 cocktail1)
)))
