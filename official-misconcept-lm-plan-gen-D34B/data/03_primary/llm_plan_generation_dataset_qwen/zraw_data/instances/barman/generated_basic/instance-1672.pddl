(define (problem prob)
 (:domain barman)
 (:objects 
      shaker270 - shaker
      left right - hand
      shot287 shot42 - shot
      ingredient216 ingredient222 ingredient289 ingredient305 - ingredient
      cocktail333 - cocktail
      dispenser488 dispenser483 dispenser240 dispenser108 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker270)
  (ontable shot287)
  (ontable shot42)
  (dispenses dispenser488 ingredient216)
  (dispenses dispenser483 ingredient222)
  (dispenses dispenser240 ingredient289)
  (dispenses dispenser108 ingredient305)
  (clean shaker270)
  (clean shot287)
  (clean shot42)
  (empty shaker270)
  (empty shot287)
  (empty shot42)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker270 l0)
  (shaker-level shaker270 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail333 ingredient222)
  (cocktail-part2 cocktail333 ingredient305)
)
 (:goal
  (and
      (contains shot287 cocktail333)
)))
