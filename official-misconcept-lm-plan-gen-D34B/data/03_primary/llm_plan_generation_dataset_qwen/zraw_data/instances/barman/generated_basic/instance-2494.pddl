(define (problem prob)
 (:domain barman)
 (:objects 
      shaker326 - shaker
      left right - hand
      shot215 shot20 shot264 - shot
      ingredient20 ingredient288 ingredient39 - ingredient
      cocktail94 - cocktail
      dispenser26 dispenser477 dispenser365 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker326)
  (ontable shot215)
  (ontable shot20)
  (ontable shot264)
  (dispenses dispenser26 ingredient20)
  (dispenses dispenser477 ingredient288)
  (dispenses dispenser365 ingredient39)
  (clean shaker326)
  (clean shot215)
  (clean shot20)
  (clean shot264)
  (empty shaker326)
  (empty shot215)
  (empty shot20)
  (empty shot264)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker326 l0)
  (shaker-level shaker326 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail94 ingredient20)
  (cocktail-part2 cocktail94 ingredient288)
)
 (:goal
  (and
      (contains shot215 cocktail94)
      (contains shot20 cocktail94)
)))
