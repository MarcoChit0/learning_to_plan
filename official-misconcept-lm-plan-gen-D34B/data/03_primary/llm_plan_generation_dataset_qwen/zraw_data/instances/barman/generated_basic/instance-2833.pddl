(define (problem prob)
 (:domain barman)
 (:objects 
      shaker97 - shaker
      left right - hand
      shot266 shot466 shot37 - shot
      ingredient12 ingredient237 ingredient64 ingredient425 - ingredient
      cocktail286 - cocktail
      dispenser222 dispenser205 dispenser489 dispenser225 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker97)
  (ontable shot266)
  (ontable shot466)
  (ontable shot37)
  (dispenses dispenser222 ingredient12)
  (dispenses dispenser205 ingredient237)
  (dispenses dispenser489 ingredient64)
  (dispenses dispenser225 ingredient425)
  (clean shaker97)
  (clean shot266)
  (clean shot466)
  (clean shot37)
  (empty shaker97)
  (empty shot266)
  (empty shot466)
  (empty shot37)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker97 l0)
  (shaker-level shaker97 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail286 ingredient425)
  (cocktail-part2 cocktail286 ingredient64)
)
 (:goal
  (and
      (contains shot266 cocktail286)
      (contains shot466 ingredient12)
)))
