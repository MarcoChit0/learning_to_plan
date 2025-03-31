(define (problem prob)
 (:domain barman)
 (:objects 
      shaker376 - shaker
      left right - hand
      shot41 shot119 - shot
      ingredient346 ingredient165 - ingredient
      cocktail1 - cocktail
      dispenser393 dispenser49 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker376)
  (ontable shot41)
  (ontable shot119)
  (dispenses dispenser393 ingredient346)
  (dispenses dispenser49 ingredient165)
  (clean shaker376)
  (clean shot41)
  (clean shot119)
  (empty shaker376)
  (empty shot41)
  (empty shot119)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker376 l0)
  (shaker-level shaker376 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient346)
  (cocktail-part2 cocktail1 ingredient165)
)
 (:goal
  (and
      (contains shot41 cocktail1)
)))
