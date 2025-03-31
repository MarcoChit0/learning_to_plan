(define (problem prob)
 (:domain barman)
 (:objects 
      shaker287 - shaker
      left right - hand
      shot205 shot450 shot266 - shot
      ingredient433 ingredient384 - ingredient
      cocktail181 - cocktail
      dispenser184 dispenser10 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker287)
  (ontable shot205)
  (ontable shot450)
  (ontable shot266)
  (dispenses dispenser184 ingredient433)
  (dispenses dispenser10 ingredient384)
  (clean shaker287)
  (clean shot205)
  (clean shot450)
  (clean shot266)
  (empty shaker287)
  (empty shot205)
  (empty shot450)
  (empty shot266)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker287 l0)
  (shaker-level shaker287 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail181 ingredient384)
  (cocktail-part2 cocktail181 ingredient433)
)
 (:goal
  (and
      (contains shot205 cocktail181)
      (contains shot450 cocktail181)
)))
