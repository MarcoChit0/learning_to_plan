(define (problem prob)
 (:domain barman)
 (:objects 
      shaker270 - shaker
      left right - hand
      shot472 shot398 shot165 - shot
      ingredient368 ingredient114 ingredient173 ingredient340 - ingredient
      cocktail138 - cocktail
      dispenser429 dispenser263 dispenser437 dispenser349 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker270)
  (ontable shot472)
  (ontable shot398)
  (ontable shot165)
  (dispenses dispenser429 ingredient368)
  (dispenses dispenser263 ingredient114)
  (dispenses dispenser437 ingredient173)
  (dispenses dispenser349 ingredient340)
  (clean shaker270)
  (clean shot472)
  (clean shot398)
  (clean shot165)
  (empty shaker270)
  (empty shot472)
  (empty shot398)
  (empty shot165)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker270 l0)
  (shaker-level shaker270 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail138 ingredient340)
  (cocktail-part2 cocktail138 ingredient114)
)
 (:goal
  (and
      (contains shot472 cocktail138)
      (contains shot398 cocktail138)
)))
