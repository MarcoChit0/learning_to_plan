(define (problem prob)
 (:domain barman)
 (:objects 
      shaker360 - shaker
      left right - hand
      shot214 shot102 shot492 - shot
      ingredient67 ingredient264 - ingredient
      cocktail405 - cocktail
      dispenser432 dispenser321 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker360)
  (ontable shot214)
  (ontable shot102)
  (ontable shot492)
  (dispenses dispenser432 ingredient67)
  (dispenses dispenser321 ingredient264)
  (clean shaker360)
  (clean shot214)
  (clean shot102)
  (clean shot492)
  (empty shaker360)
  (empty shot214)
  (empty shot102)
  (empty shot492)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker360 l0)
  (shaker-level shaker360 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail405 ingredient67)
  (cocktail-part2 cocktail405 ingredient264)
)
 (:goal
  (and
      (contains shot214 cocktail405)
      (contains shot102 cocktail405)
)))
