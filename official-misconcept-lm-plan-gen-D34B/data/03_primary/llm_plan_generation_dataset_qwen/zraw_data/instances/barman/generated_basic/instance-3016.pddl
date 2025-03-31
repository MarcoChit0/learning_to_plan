(define (problem prob)
 (:domain barman)
 (:objects 
      shaker429 - shaker
      left right - hand
      shot377 shot264 - shot
      ingredient11 ingredient304 ingredient426 - ingredient
      cocktail376 - cocktail
      dispenser443 dispenser218 dispenser378 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker429)
  (ontable shot377)
  (ontable shot264)
  (dispenses dispenser443 ingredient11)
  (dispenses dispenser218 ingredient304)
  (dispenses dispenser378 ingredient426)
  (clean shaker429)
  (clean shot377)
  (clean shot264)
  (empty shaker429)
  (empty shot377)
  (empty shot264)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker429 l0)
  (shaker-level shaker429 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail376 ingredient304)
  (cocktail-part2 cocktail376 ingredient426)
)
 (:goal
  (and
      (contains shot377 cocktail376)
)))
