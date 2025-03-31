(define (problem prob)
 (:domain barman)
 (:objects 
      shaker77 - shaker
      left right - hand
      shot368 shot446 - shot
      ingredient175 ingredient19 ingredient370 - ingredient
      cocktail162 - cocktail
      dispenser424 dispenser314 dispenser165 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker77)
  (ontable shot368)
  (ontable shot446)
  (dispenses dispenser424 ingredient175)
  (dispenses dispenser314 ingredient19)
  (dispenses dispenser165 ingredient370)
  (clean shaker77)
  (clean shot368)
  (clean shot446)
  (empty shaker77)
  (empty shot368)
  (empty shot446)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker77 l0)
  (shaker-level shaker77 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail162 ingredient175)
  (cocktail-part2 cocktail162 ingredient370)
)
 (:goal
  (and
      (contains shot368 cocktail162)
)))
