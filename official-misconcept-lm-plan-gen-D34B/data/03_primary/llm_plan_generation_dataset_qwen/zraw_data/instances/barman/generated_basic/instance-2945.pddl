(define (problem prob)
 (:domain barman)
 (:objects 
      shaker456 - shaker
      left right - hand
      shot394 - shot
      ingredient280 ingredient418 ingredient470 ingredient263 - ingredient
      cocktail241 - cocktail
      dispenser426 dispenser462 dispenser229 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker456)
  (ontable shot394)
  (dispenses dispenser426 ingredient280)
  (dispenses dispenser462 ingredient418)
  (dispenses dispenser229 ingredient470)
  (dispenses dispenser92 ingredient263)
  (clean shaker456)
  (clean shot394)
  (empty shaker456)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker456 l0)
  (shaker-level shaker456 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail241 ingredient418)
  (cocktail-part2 cocktail241 ingredient280)
)
 (:goal
  (and
      (contains shot394 cocktail241)
)))
