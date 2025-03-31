(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot149 shot238 shot484 - shot
      ingredient55 ingredient344 ingredient397 ingredient15 - ingredient
      cocktail181 - cocktail
      dispenser126 dispenser102 dispenser44 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot149)
  (ontable shot238)
  (ontable shot484)
  (dispenses dispenser126 ingredient55)
  (dispenses dispenser102 ingredient344)
  (dispenses dispenser44 ingredient397)
  (dispenses dispenser492 ingredient15)
  (clean shaker262)
  (clean shot149)
  (clean shot238)
  (clean shot484)
  (empty shaker262)
  (empty shot149)
  (empty shot238)
  (empty shot484)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail181 ingredient344)
  (cocktail-part2 cocktail181 ingredient397)
)
 (:goal
  (and
      (contains shot149 cocktail181)
      (contains shot238 cocktail181)
)))
