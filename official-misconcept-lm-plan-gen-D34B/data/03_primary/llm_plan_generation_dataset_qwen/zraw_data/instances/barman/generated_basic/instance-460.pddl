(define (problem prob)
 (:domain barman)
 (:objects 
      shaker57 - shaker
      left right - hand
      shot8 shot360 shot44 - shot
      ingredient28 ingredient89 ingredient328 ingredient357 - ingredient
      cocktail1 - cocktail
      dispenser464 dispenser175 dispenser341 dispenser105 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker57)
  (ontable shot8)
  (ontable shot360)
  (ontable shot44)
  (dispenses dispenser464 ingredient28)
  (dispenses dispenser175 ingredient89)
  (dispenses dispenser341 ingredient328)
  (dispenses dispenser105 ingredient357)
  (clean shaker57)
  (clean shot8)
  (clean shot360)
  (clean shot44)
  (empty shaker57)
  (empty shot8)
  (empty shot360)
  (empty shot44)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker57 l0)
  (shaker-level shaker57 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient28)
  (cocktail-part2 cocktail1 ingredient357)
)
 (:goal
  (and
      (contains shot8 cocktail1)
      (contains shot360 cocktail1)
)))
