(define (problem prob)
 (:domain barman)
 (:objects 
      shaker388 - shaker
      left right - hand
      shot40 shot443 - shot
      ingredient193 ingredient89 - ingredient
      cocktail343 - cocktail
      dispenser1 dispenser249 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker388)
  (ontable shot40)
  (ontable shot443)
  (dispenses dispenser1 ingredient193)
  (dispenses dispenser249 ingredient89)
  (clean shaker388)
  (clean shot40)
  (clean shot443)
  (empty shaker388)
  (empty shot40)
  (empty shot443)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker388 l0)
  (shaker-level shaker388 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail343 ingredient193)
  (cocktail-part2 cocktail343 ingredient89)
)
 (:goal
  (and
      (contains shot40 cocktail343)
)))
