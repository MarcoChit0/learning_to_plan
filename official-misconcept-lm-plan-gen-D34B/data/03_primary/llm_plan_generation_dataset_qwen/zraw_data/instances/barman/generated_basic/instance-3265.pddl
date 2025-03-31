(define (problem prob)
 (:domain barman)
 (:objects 
      shaker83 - shaker
      left right - hand
      shot225 shot428 shot434 - shot
      ingredient17 ingredient462 ingredient108 ingredient378 - ingredient
      cocktail76 - cocktail
      dispenser20 dispenser490 dispenser167 dispenser359 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker83)
  (ontable shot225)
  (ontable shot428)
  (ontable shot434)
  (dispenses dispenser20 ingredient17)
  (dispenses dispenser490 ingredient462)
  (dispenses dispenser167 ingredient108)
  (dispenses dispenser359 ingredient378)
  (clean shaker83)
  (clean shot225)
  (clean shot428)
  (clean shot434)
  (empty shaker83)
  (empty shot225)
  (empty shot428)
  (empty shot434)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker83 l0)
  (shaker-level shaker83 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail76 ingredient17)
  (cocktail-part2 cocktail76 ingredient462)
)
 (:goal
  (and
      (contains shot225 cocktail76)
      (contains shot428 cocktail76)
)))
