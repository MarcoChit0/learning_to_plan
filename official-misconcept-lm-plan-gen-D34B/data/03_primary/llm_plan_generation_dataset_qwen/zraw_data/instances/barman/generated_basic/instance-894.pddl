(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot8 shot293 - shot
      ingredient134 ingredient77 - ingredient
      cocktail1 - cocktail
      dispenser398 dispenser138 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot8)
  (ontable shot293)
  (dispenses dispenser398 ingredient134)
  (dispenses dispenser138 ingredient77)
  (clean shaker45)
  (clean shot8)
  (clean shot293)
  (empty shaker45)
  (empty shot8)
  (empty shot293)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient77)
  (cocktail-part2 cocktail1 ingredient134)
)
 (:goal
  (and
      (contains shot8 cocktail1)
)))
