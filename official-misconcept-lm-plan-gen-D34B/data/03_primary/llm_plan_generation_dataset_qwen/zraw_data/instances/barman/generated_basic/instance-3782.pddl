(define (problem prob)
 (:domain barman)
 (:objects 
      shaker71 - shaker
      left right - hand
      shot331 shot264 shot346 - shot
      ingredient260 ingredient402 - ingredient
      cocktail288 - cocktail
      dispenser81 dispenser378 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker71)
  (ontable shot331)
  (ontable shot264)
  (ontable shot346)
  (dispenses dispenser81 ingredient260)
  (dispenses dispenser378 ingredient402)
  (clean shaker71)
  (clean shot331)
  (clean shot264)
  (clean shot346)
  (empty shaker71)
  (empty shot331)
  (empty shot264)
  (empty shot346)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker71 l0)
  (shaker-level shaker71 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail288 ingredient260)
  (cocktail-part2 cocktail288 ingredient402)
)
 (:goal
  (and
      (contains shot331 cocktail288)
      (contains shot264 cocktail288)
)))
