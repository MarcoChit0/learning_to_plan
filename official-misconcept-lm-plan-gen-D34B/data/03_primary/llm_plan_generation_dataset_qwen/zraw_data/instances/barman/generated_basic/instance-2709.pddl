(define (problem prob)
 (:domain barman)
 (:objects 
      shaker481 - shaker
      left right - hand
      shot134 shot166 shot261 - shot
      ingredient146 ingredient372 - ingredient
      cocktail170 - cocktail
      dispenser300 dispenser283 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker481)
  (ontable shot134)
  (ontable shot166)
  (ontable shot261)
  (dispenses dispenser300 ingredient146)
  (dispenses dispenser283 ingredient372)
  (clean shaker481)
  (clean shot134)
  (clean shot166)
  (clean shot261)
  (empty shaker481)
  (empty shot134)
  (empty shot166)
  (empty shot261)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker481 l0)
  (shaker-level shaker481 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail170 ingredient372)
  (cocktail-part2 cocktail170 ingredient146)
)
 (:goal
  (and
      (contains shot134 cocktail170)
      (contains shot166 ingredient372)
)))
