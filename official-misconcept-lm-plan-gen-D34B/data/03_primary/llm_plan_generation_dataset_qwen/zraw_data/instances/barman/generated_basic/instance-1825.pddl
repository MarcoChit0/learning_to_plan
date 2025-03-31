(define (problem prob)
 (:domain barman)
 (:objects 
      shaker207 - shaker
      left right - hand
      shot384 shot76 shot248 - shot
      ingredient267 ingredient260 ingredient65 ingredient165 - ingredient
      cocktail264 - cocktail
      dispenser492 dispenser419 dispenser409 dispenser54 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker207)
  (ontable shot384)
  (ontable shot76)
  (ontable shot248)
  (dispenses dispenser492 ingredient267)
  (dispenses dispenser419 ingredient260)
  (dispenses dispenser409 ingredient65)
  (dispenses dispenser54 ingredient165)
  (clean shaker207)
  (clean shot384)
  (clean shot76)
  (clean shot248)
  (empty shaker207)
  (empty shot384)
  (empty shot76)
  (empty shot248)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker207 l0)
  (shaker-level shaker207 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail264 ingredient165)
  (cocktail-part2 cocktail264 ingredient65)
)
 (:goal
  (and
      (contains shot384 cocktail264)
      (contains shot76 ingredient260)
)))
