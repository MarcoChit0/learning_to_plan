(define (problem prob)
 (:domain barman)
 (:objects 
      shaker339 - shaker
      left right - hand
      shot323 shot50 - shot
      ingredient50 ingredient2 ingredient256 ingredient436 - ingredient
      cocktail1 - cocktail
      dispenser88 dispenser482 dispenser332 dispenser377 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker339)
  (ontable shot323)
  (ontable shot50)
  (dispenses dispenser88 ingredient50)
  (dispenses dispenser482 ingredient2)
  (dispenses dispenser332 ingredient256)
  (dispenses dispenser377 ingredient436)
  (clean shaker339)
  (clean shot323)
  (clean shot50)
  (empty shaker339)
  (empty shot323)
  (empty shot50)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker339 l0)
  (shaker-level shaker339 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient50)
  (cocktail-part2 cocktail1 ingredient256)
)
 (:goal
  (and
      (contains shot323 cocktail1)
)))
