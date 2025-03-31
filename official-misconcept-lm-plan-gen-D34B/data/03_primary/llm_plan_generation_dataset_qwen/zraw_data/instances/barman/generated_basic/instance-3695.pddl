(define (problem prob)
 (:domain barman)
 (:objects 
      shaker60 - shaker
      left right - hand
      shot392 shot181 - shot
      ingredient255 ingredient393 ingredient257 - ingredient
      cocktail409 - cocktail
      dispenser100 dispenser351 dispenser12 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker60)
  (ontable shot392)
  (ontable shot181)
  (dispenses dispenser100 ingredient255)
  (dispenses dispenser351 ingredient393)
  (dispenses dispenser12 ingredient257)
  (clean shaker60)
  (clean shot392)
  (clean shot181)
  (empty shaker60)
  (empty shot392)
  (empty shot181)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker60 l0)
  (shaker-level shaker60 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail409 ingredient393)
  (cocktail-part2 cocktail409 ingredient255)
)
 (:goal
  (and
      (contains shot392 cocktail409)
)))
