(define (problem prob)
 (:domain barman)
 (:objects 
      shaker210 - shaker
      left right - hand
      shot204 shot492 - shot
      ingredient192 ingredient264 ingredient271 ingredient399 - ingredient
      cocktail412 - cocktail
      dispenser418 dispenser408 dispenser56 dispenser299 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker210)
  (ontable shot204)
  (ontable shot492)
  (dispenses dispenser418 ingredient192)
  (dispenses dispenser408 ingredient264)
  (dispenses dispenser56 ingredient271)
  (dispenses dispenser299 ingredient399)
  (clean shaker210)
  (clean shot204)
  (clean shot492)
  (empty shaker210)
  (empty shot204)
  (empty shot492)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker210 l0)
  (shaker-level shaker210 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail412 ingredient399)
  (cocktail-part2 cocktail412 ingredient264)
)
 (:goal
  (and
      (contains shot204 cocktail412)
)))
