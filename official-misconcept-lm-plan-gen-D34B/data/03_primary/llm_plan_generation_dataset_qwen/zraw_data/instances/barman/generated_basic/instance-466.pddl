(define (problem prob)
 (:domain barman)
 (:objects 
      shaker44 - shaker
      left right - hand
      shot314 shot237 shot299 - shot
      ingredient423 ingredient12 ingredient87 ingredient137 - ingredient
      cocktail1 - cocktail
      dispenser460 dispenser183 dispenser61 dispenser119 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker44)
  (ontable shot314)
  (ontable shot237)
  (ontable shot299)
  (dispenses dispenser460 ingredient423)
  (dispenses dispenser183 ingredient12)
  (dispenses dispenser61 ingredient87)
  (dispenses dispenser119 ingredient137)
  (clean shaker44)
  (clean shot314)
  (clean shot237)
  (clean shot299)
  (empty shaker44)
  (empty shot314)
  (empty shot237)
  (empty shot299)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker44 l0)
  (shaker-level shaker44 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient12)
  (cocktail-part2 cocktail1 ingredient137)
)
 (:goal
  (and
      (contains shot314 cocktail1)
      (contains shot237 cocktail1)
)))
