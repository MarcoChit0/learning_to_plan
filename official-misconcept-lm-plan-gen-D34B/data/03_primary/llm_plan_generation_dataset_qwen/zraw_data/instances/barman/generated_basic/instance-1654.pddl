(define (problem prob)
 (:domain barman)
 (:objects 
      shaker317 - shaker
      left right - hand
      shot287 shot109 shot405 - shot
      ingredient200 ingredient454 ingredient148 ingredient380 - ingredient
      cocktail114 - cocktail
      dispenser290 dispenser412 dispenser469 dispenser459 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker317)
  (ontable shot287)
  (ontable shot109)
  (ontable shot405)
  (dispenses dispenser290 ingredient200)
  (dispenses dispenser412 ingredient454)
  (dispenses dispenser469 ingredient148)
  (dispenses dispenser459 ingredient380)
  (clean shaker317)
  (clean shot287)
  (clean shot109)
  (clean shot405)
  (empty shaker317)
  (empty shot287)
  (empty shot109)
  (empty shot405)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker317 l0)
  (shaker-level shaker317 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail114 ingredient148)
  (cocktail-part2 cocktail114 ingredient200)
)
 (:goal
  (and
      (contains shot287 cocktail114)
      (contains shot109 cocktail114)
)))
