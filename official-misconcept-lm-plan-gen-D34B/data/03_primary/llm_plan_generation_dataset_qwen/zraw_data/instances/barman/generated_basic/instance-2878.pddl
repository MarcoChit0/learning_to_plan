(define (problem prob)
 (:domain barman)
 (:objects 
      shaker449 - shaker
      left right - hand
      shot136 shot258 shot145 - shot
      ingredient202 ingredient81 - ingredient
      cocktail227 - cocktail
      dispenser308 dispenser383 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker449)
  (ontable shot136)
  (ontable shot258)
  (ontable shot145)
  (dispenses dispenser308 ingredient202)
  (dispenses dispenser383 ingredient81)
  (clean shaker449)
  (clean shot136)
  (clean shot258)
  (clean shot145)
  (empty shaker449)
  (empty shot136)
  (empty shot258)
  (empty shot145)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker449 l0)
  (shaker-level shaker449 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail227 ingredient202)
  (cocktail-part2 cocktail227 ingredient81)
)
 (:goal
  (and
      (contains shot136 cocktail227)
      (contains shot258 cocktail227)
)))
