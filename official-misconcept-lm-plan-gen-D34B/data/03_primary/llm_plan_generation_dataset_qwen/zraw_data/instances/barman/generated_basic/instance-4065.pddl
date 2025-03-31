(define (problem prob)
 (:domain barman)
 (:objects 
      shaker430 - shaker
      left right - hand
      shot20 shot223 - shot
      ingredient68 ingredient434 - ingredient
      cocktail231 - cocktail
      dispenser473 dispenser428 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker430)
  (ontable shot20)
  (ontable shot223)
  (dispenses dispenser473 ingredient68)
  (dispenses dispenser428 ingredient434)
  (clean shaker430)
  (clean shot20)
  (clean shot223)
  (empty shaker430)
  (empty shot20)
  (empty shot223)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker430 l0)
  (shaker-level shaker430 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail231 ingredient434)
  (cocktail-part2 cocktail231 ingredient68)
)
 (:goal
  (and
      (contains shot20 cocktail231)
)))
