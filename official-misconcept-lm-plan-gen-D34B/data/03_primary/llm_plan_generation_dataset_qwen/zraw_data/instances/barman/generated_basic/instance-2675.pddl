(define (problem prob)
 (:domain barman)
 (:objects 
      shaker119 - shaker
      left right - hand
      shot89 shot490 - shot
      ingredient328 ingredient394 ingredient303 - ingredient
      cocktail111 - cocktail
      dispenser404 dispenser256 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker119)
  (ontable shot89)
  (ontable shot490)
  (dispenses dispenser404 ingredient328)
  (dispenses dispenser256 ingredient394)
  (dispenses dispenser207 ingredient303)
  (clean shaker119)
  (clean shot89)
  (clean shot490)
  (empty shaker119)
  (empty shot89)
  (empty shot490)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker119 l0)
  (shaker-level shaker119 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail111 ingredient303)
  (cocktail-part2 cocktail111 ingredient394)
)
 (:goal
  (and
      (contains shot89 cocktail111)
)))
