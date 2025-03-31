(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot148 shot109 shot429 - shot
      ingredient20 ingredient47 ingredient441 ingredient301 - ingredient
      cocktail418 - cocktail
      dispenser160 dispenser357 dispenser33 dispenser253 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot148)
  (ontable shot109)
  (ontable shot429)
  (dispenses dispenser160 ingredient20)
  (dispenses dispenser357 ingredient47)
  (dispenses dispenser33 ingredient441)
  (dispenses dispenser253 ingredient301)
  (clean shaker221)
  (clean shot148)
  (clean shot109)
  (clean shot429)
  (empty shaker221)
  (empty shot148)
  (empty shot109)
  (empty shot429)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail418 ingredient441)
  (cocktail-part2 cocktail418 ingredient47)
)
 (:goal
  (and
      (contains shot148 cocktail418)
      (contains shot109 ingredient20)
)))
