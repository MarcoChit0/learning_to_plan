(define (problem prob)
 (:domain barman)
 (:objects 
      shaker210 - shaker
      left right - hand
      shot399 shot294 - shot
      ingredient188 ingredient45 ingredient454 ingredient11 - ingredient
      cocktail317 - cocktail
      dispenser271 dispenser139 dispenser251 dispenser275 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker210)
  (ontable shot399)
  (ontable shot294)
  (dispenses dispenser271 ingredient188)
  (dispenses dispenser139 ingredient45)
  (dispenses dispenser251 ingredient454)
  (dispenses dispenser275 ingredient11)
  (clean shaker210)
  (clean shot399)
  (clean shot294)
  (empty shaker210)
  (empty shot399)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker210 l0)
  (shaker-level shaker210 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail317 ingredient11)
  (cocktail-part2 cocktail317 ingredient45)
)
 (:goal
  (and
      (contains shot399 cocktail317)
)))
