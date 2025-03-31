(define (problem prob)
 (:domain barman)
 (:objects 
      shaker287 - shaker
      left right - hand
      shot468 shot420 shot369 - shot
      ingredient104 ingredient120 ingredient357 - ingredient
      cocktail35 - cocktail
      dispenser428 dispenser235 dispenser191 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker287)
  (ontable shot468)
  (ontable shot420)
  (ontable shot369)
  (dispenses dispenser428 ingredient104)
  (dispenses dispenser235 ingredient120)
  (dispenses dispenser191 ingredient357)
  (clean shaker287)
  (clean shot468)
  (clean shot420)
  (clean shot369)
  (empty shaker287)
  (empty shot468)
  (empty shot420)
  (empty shot369)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker287 l0)
  (shaker-level shaker287 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail35 ingredient104)
  (cocktail-part2 cocktail35 ingredient357)
)
 (:goal
  (and
      (contains shot468 cocktail35)
      (contains shot420 cocktail35)
)))
