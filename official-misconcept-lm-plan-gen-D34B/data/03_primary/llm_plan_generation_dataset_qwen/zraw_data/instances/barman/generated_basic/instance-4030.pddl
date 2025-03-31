(define (problem prob)
 (:domain barman)
 (:objects 
      shaker146 - shaker
      left right - hand
      shot141 shot460 - shot
      ingredient115 ingredient342 ingredient222 - ingredient
      cocktail316 - cocktail
      dispenser32 dispenser266 dispenser387 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker146)
  (ontable shot141)
  (ontable shot460)
  (dispenses dispenser32 ingredient115)
  (dispenses dispenser266 ingredient342)
  (dispenses dispenser387 ingredient222)
  (clean shaker146)
  (clean shot141)
  (clean shot460)
  (empty shaker146)
  (empty shot141)
  (empty shot460)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker146 l0)
  (shaker-level shaker146 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail316 ingredient115)
  (cocktail-part2 cocktail316 ingredient342)
)
 (:goal
  (and
      (contains shot141 cocktail316)
)))
