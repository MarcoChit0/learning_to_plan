(define (problem prob)
 (:domain barman)
 (:objects 
      shaker285 - shaker
      left right - hand
      shot481 shot134 shot364 - shot
      ingredient303 ingredient458 ingredient82 ingredient102 - ingredient
      cocktail1 - cocktail
      dispenser143 dispenser176 dispenser150 dispenser118 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker285)
  (ontable shot481)
  (ontable shot134)
  (ontable shot364)
  (dispenses dispenser143 ingredient303)
  (dispenses dispenser176 ingredient458)
  (dispenses dispenser150 ingredient82)
  (dispenses dispenser118 ingredient102)
  (clean shaker285)
  (clean shot481)
  (clean shot134)
  (clean shot364)
  (empty shaker285)
  (empty shot481)
  (empty shot134)
  (empty shot364)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker285 l0)
  (shaker-level shaker285 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient102)
  (cocktail-part2 cocktail1 ingredient458)
)
 (:goal
  (and
      (contains shot481 cocktail1)
      (contains shot134 ingredient303)
)))
