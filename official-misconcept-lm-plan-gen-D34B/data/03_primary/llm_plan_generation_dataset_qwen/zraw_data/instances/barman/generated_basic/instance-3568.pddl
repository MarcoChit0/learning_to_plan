(define (problem prob)
 (:domain barman)
 (:objects 
      shaker181 - shaker
      left right - hand
      shot314 shot367 - shot
      ingredient133 ingredient375 ingredient435 ingredient268 - ingredient
      cocktail293 - cocktail
      dispenser398 dispenser160 dispenser259 dispenser389 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker181)
  (ontable shot314)
  (ontable shot367)
  (dispenses dispenser398 ingredient133)
  (dispenses dispenser160 ingredient375)
  (dispenses dispenser259 ingredient435)
  (dispenses dispenser389 ingredient268)
  (clean shaker181)
  (clean shot314)
  (clean shot367)
  (empty shaker181)
  (empty shot314)
  (empty shot367)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker181 l0)
  (shaker-level shaker181 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail293 ingredient435)
  (cocktail-part2 cocktail293 ingredient133)
)
 (:goal
  (and
      (contains shot314 cocktail293)
)))
