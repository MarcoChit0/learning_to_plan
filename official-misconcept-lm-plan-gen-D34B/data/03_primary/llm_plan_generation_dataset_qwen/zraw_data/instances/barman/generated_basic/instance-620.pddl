(define (problem prob)
 (:domain barman)
 (:objects 
      shaker287 - shaker
      left right - hand
      shot249 shot149 - shot
      ingredient192 ingredient416 ingredient437 ingredient4 - ingredient
      cocktail1 - cocktail
      dispenser61 dispenser293 dispenser420 dispenser341 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker287)
  (ontable shot249)
  (ontable shot149)
  (dispenses dispenser61 ingredient192)
  (dispenses dispenser293 ingredient416)
  (dispenses dispenser420 ingredient437)
  (dispenses dispenser341 ingredient4)
  (clean shaker287)
  (clean shot249)
  (clean shot149)
  (empty shaker287)
  (empty shot249)
  (empty shot149)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker287 l0)
  (shaker-level shaker287 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient192)
  (cocktail-part2 cocktail1 ingredient416)
)
 (:goal
  (and
      (contains shot249 cocktail1)
)))
