(define (problem prob)
 (:domain barman)
 (:objects 
      shaker287 - shaker
      left right - hand
      shot67 shot143 shot114 - shot
      ingredient71 ingredient486 ingredient196 ingredient440 - ingredient
      cocktail342 - cocktail
      dispenser48 dispenser61 dispenser71 dispenser262 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker287)
  (ontable shot67)
  (ontable shot143)
  (ontable shot114)
  (dispenses dispenser48 ingredient71)
  (dispenses dispenser61 ingredient486)
  (dispenses dispenser71 ingredient196)
  (dispenses dispenser262 ingredient440)
  (clean shaker287)
  (clean shot67)
  (clean shot143)
  (clean shot114)
  (empty shaker287)
  (empty shot67)
  (empty shot143)
  (empty shot114)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker287 l0)
  (shaker-level shaker287 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail342 ingredient440)
  (cocktail-part2 cocktail342 ingredient196)
)
 (:goal
  (and
      (contains shot67 cocktail342)
      (contains shot143 ingredient486)
)))
